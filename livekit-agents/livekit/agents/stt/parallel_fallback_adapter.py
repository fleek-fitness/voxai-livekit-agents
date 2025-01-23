from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from dataclasses import dataclass
from typing import Literal

from livekit import rtc
from livekit.agents.utils.audio import AudioBuffer

from .. import utils
from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

# don't retry when using the fallback adapter
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


@dataclass
class _STTStatus:
    available: bool
    recovering_synthesize_task: asyncio.Task | None
    recovering_stream_task: asyncio.Task | None


class ParallelFallbackAdapter(
    STT[Literal["stt_availability_changed"]],
):
    """
    Modified FallbackAdapter that runs multiple STTs *in parallel*. After collecting
    all their results (or exceptions/timeouts), it picks the final transcript in
    *priority order*:
        1) stt0's result (if valid & non-empty)
        2) stt1's result (if valid & non-empty)
        3) etc.

    If none gave a valid transcript, it raises an APIConnectionError.
    """

    def __init__(
        self,
        stt: list[STT],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        if len(stt) < 1:
            raise ValueError("At least one STT instance must be provided.")

        # We require that *all* STT have streaming if we want streaming fallback:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=all(t.capabilities.streaming for t in stt),
                interim_results=all(t.capabilities.interim_results for t in stt),
            )
        )

        self._stt_instances = stt
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval

        self._status: list[_STTStatus] = [
            _STTStatus(
                available=True,
                recovering_synthesize_task=None,
                recovering_stream_task=None,
            )
            for _ in self._stt_instances
        ]

    async def _try_recognize(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: str | None = None,
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> SpeechEvent:
        """
        Performs a single STT's recognize() call with the given buffer.
        """
        try:
            return await stt.recognize(
                buffer,
                language=language,
                conn_options=dataclasses.replace(
                    conn_options,
                    max_retry=self._max_retry_per_stt,
                    timeout=self._attempt_timeout,
                    retry_interval=self._retry_interval,
                ),
            )
        except asyncio.TimeoutError:
            if recovering:
                logger.warning(
                    f"{stt.label} recovery timed out", extra={"streamed": False}
                )
                raise

            logger.warning(
                f"{stt.label} timed out, switching to next STT",
                extra={"streamed": False},
            )
            raise
        except APIError as e:
            if recovering:
                logger.warning(
                    f"{stt.label} recovery failed",
                    exc_info=e,
                    extra={"streamed": False},
                )
                raise

            logger.warning(
                f"{stt.label} failed, switching to next STT",
                exc_info=e,
                extra={"streamed": False},
            )
            raise
        except Exception:
            if recovering:
                logger.exception(
                    f"{stt.label} recovery unexpected error", extra={"streamed": False}
                )
                raise

            logger.exception(
                f"{stt.label} unexpected error, switching to next STT",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        """
        Attempts to recover an STT that just failed, by spawning a background recognize
        task with the same data. If that recovers successfully, we mark it as available again.
        """
        stt_status = self._status[self._stt_instances.index(stt)]
        if (
            stt_status.recovering_synthesize_task is None
            or stt_status.recovering_synthesize_task.done()
        ):

            async def _recover_stt_task(stt: STT) -> None:
                try:
                    await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=True,
                    )
                    # If the above doesn't raise, we can mark STT as recovered
                    stt_status.available = True
                    logger.info(f"{stt.label} recovered")
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )
                except Exception:
                    # If we still fail, remain unavailable
                    return

            stt_status.recovering_synthesize_task = asyncio.create_task(
                _recover_stt_task(stt)
            )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        """
        Launches *all* STT in parallel. Then picks the final transcript
        in priority order. If the highest priority that has a non-empty transcript
        is found, we return that. Otherwise keep going. If all are empty or fail,
        raise APIConnectionError.
        """

        start_time = time.time()
        statuses = self._status

        # If all are unavailable, we'll still attempt them to see if they recover:
        all_failed_before = all(not s.available for s in statuses)
        if all_failed_before:
            logger.error("all STTs are unavailable, retrying..")

        # STEP 1: Create tasks for each STT that is "available" or if all were failed (we try anyway).
        tasks = []
        stts_involved = []
        for i, stt in enumerate(self._stt_instances):
            stt_status = statuses[i]
            if stt_status.available or all_failed_before:
                # We try it in parallel
                task = asyncio.create_task(
                    self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=False,
                    )
                )
                tasks.append(task)
                stts_involved.append(i)
            else:
                # We'll attempt to spawn a recovery in background
                self._try_recovery(
                    stt=stt, buffer=buffer, language=language, conn_options=conn_options
                )

        # If we have no tasks at all to run, then everything is unavailable and can't even be tried
        if not tasks:
            raise APIConnectionError(
                "all STTs are permanently unavailable, cannot proceed."
            )

        # STEP 2: Gather results (some may fail with exceptions)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # STEP 3: Evaluate results in *priority order*. That means we check stts_involved
        # in ascending order, which matches the order in which they appear in self._stt_instances.
        # The first non-empty transcript we get from the highest priority is returned.
        best_event = None
        best_index = None

        for idx, res in zip(stts_involved, results):
            stt = self._stt_instances[idx]
            if isinstance(res, Exception):
                # Mark STT as unavailable, start recovery, skip
                if statuses[idx].available:
                    statuses[idx].available = False
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=False),
                    )
                # Attempt recovery in background
                self._try_recovery(
                    stt=stt, buffer=buffer, language=language, conn_options=conn_options
                )
                continue

            # If we got a successful SpeechEvent, check if it has text
            event = res  # type: SpeechEvent
            # Here you can define "non-empty" however you wish:
            if event.alternatives and event.alternatives[0].text.strip():
                # We'll pick the first (highest priority) that has text
                best_event = event
                best_index = idx
                break
            else:
                # The STT responded but it's empty, so we treat that as a "failure" for fallback
                if statuses[idx].available:
                    statuses[idx].available = False
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=False),
                    )
                self._try_recovery(
                    stt=stt, buffer=buffer, language=language, conn_options=conn_options
                )

        if best_event is None:
            # No STT returned anything non-empty
            raise APIConnectionError(
                "All STTs returned empty or failed (%s) after %.2f seconds"
                % (
                    [stt.label for stt in self._stt_instances],
                    time.time() - start_time,
                )
            )

        return best_event

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        # The public-facing method
        return await super().recognize(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(
            stt=self, language=language, conn_options=conn_options
        )

    async def aclose(self) -> None:
        # Cancel any background recovery tasks
        for stt_status in self._status:
            if stt_status.recovering_synthesize_task is not None:
                await aio.gracefully_cancel(stt_status.recovering_synthesize_task)
            if stt_status.recovering_stream_task is not None:
                await aio.gracefully_cancel(stt_status.recovering_stream_task)


class FallbackRecognizeStream(RecognizeStream):
    """
    Parallel streaming version. We open multiple streams in parallel, and
    feed audio frames to all of them. We capture whichever results come in.
    Once we see a non-empty final transcript from the highest priority STT
    that eventually yields results, we use it.

    If multiple STTs finish, we prefer the top of the list (self._stt_instances[0]),
    then the next, etc. If they are empty or fail, we fallback further.
    """

    def __init__(
        self,
        *,
        stt: ParallelFallbackAdapter,
        language: str | None,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=None)
        self._language = language
        self._fallback_adapter = stt
        self._recovering_streams: list[RecognizeStream] = []
        self._active_streams: list[RecognizeStream] = []
        self._forward_input_task: asyncio.Task | None = None

    async def _cleanup_streams(self) -> None:
        """Clean up all active and recovery streams."""
        # Cancel forward input task if it exists
        if self._forward_input_task and not self._forward_input_task.done():
            await aio.gracefully_cancel(self._forward_input_task)
            self._forward_input_task = None

        # Close all active streams
        for stream in self._active_streams:
            try:
                await stream.aclose()
            except Exception:
                pass
        self._active_streams.clear()

        # Close all recovery streams
        for stream in self._recovering_streams:
            try:
                await stream.aclose()
            except Exception:
                pass
        self._recovering_streams.clear()

    async def aclose(self) -> None:
        """Ensure proper cleanup when the stream is closed."""
        await self._cleanup_streams()
        await super().aclose()

    async def _run(self) -> None:
        try:
            start_time = time.time()

            # Clean up any existing streams before starting new ones
            await self._cleanup_streams()

            # If all STTs are unavailable, we still attempt them (they may recover)
            statuses = self._fallback_adapter._status
            all_failed_before = all(not s.available for s in statuses)
            if all_failed_before:
                logger.error("all STTs are unavailable, retrying..")

            # Create parallel streams for those that are available
            stt_indices = []
            for i, stt in enumerate(self._fallback_adapter._stt_instances):
                stt_status = statuses[i]
                if stt_status.available or all_failed_before:
                    stream = stt.stream(
                        language=self._language,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=self._fallback_adapter._max_retry_per_stt,
                            timeout=self._fallback_adapter._attempt_timeout,
                            retry_interval=self._fallback_adapter._retry_interval,
                        ),
                    )
                    self._active_streams.append(stream)
                    stt_indices.append(i)
                else:
                    self._try_recovery(stt)

            if not self._active_streams:
                raise APIConnectionError(
                    "all STTs are permanently unavailable, cannot stream"
                )

            # Start forwarding input to all streams
            self._forward_input_task = asyncio.create_task(
                self._forward_input_to_streams(self._active_streams)
            )

            # Create consumption tasks
            consumption_tasks = [
                asyncio.create_task(self._consume_stream(i, s))
                for i, s in zip(stt_indices, self._active_streams)
            ]

            # Wait for all tasks to complete
            await asyncio.wait(consumption_tasks, return_when=asyncio.ALL_COMPLETED)

            # Process results
            best_event = None
            best_index = None

            for idx, task in zip(stt_indices, consumption_tasks):
                stt = self._fallback_adapter._stt_instances[idx]
                if task.done():
                    try:
                        result = task.result()
                        if isinstance(result, SpeechEvent):
                            if (
                                result.alternatives
                                and result.alternatives[0].text.strip()
                            ):
                                best_event = result
                                best_index = idx
                                break
                    except Exception as e:
                        if statuses[idx].available:
                            statuses[idx].available = False
                            self._fallback_adapter.emit(
                                "stt_availability_changed",
                                AvailabilityChangedEvent(stt=stt, available=False),
                            )
                        self._try_recovery(stt)

            if best_event is None:
                raise APIConnectionError(
                    "all STTs failed or returned empty (%s) after %.2f seconds"
                    % (
                        [stt.label for stt in self._fallback_adapter._stt_instances],
                        time.time() - start_time,
                    )
                )

            self._event_ch.send_nowait(best_event)

        finally:
            # Always clean up streams when done
            await self._cleanup_streams()

    async def _forward_input_to_streams(self, streams: list[RecognizeStream]) -> None:
        """Reads from self._input_ch and forwards audio frames to each stream in parallel."""
        try:
            async for data in self._input_ch:
                for stream in streams:
                    if isinstance(data, rtc.AudioFrame):
                        stream.push_frame(data)
                    elif isinstance(data, self._FlushSentinel):
                        stream.flush()
            for stream in streams:
                stream.end_input()
        except Exception:
            # Suppress errors during cleanup
            pass

    async def _consume_stream(self, idx: int, stream: RecognizeStream):
        """
        Collects events from the given stream. If we get a final transcript event
        with non-empty text, we return that. If the stream ends, we return the
        *last* final transcript or None. Raises on errors.
        """
        stt = self._fallback_adapter._stt_instances[idx]
        try:
            async with stream:
                async for ev in stream:
                    # Forward *all* events up the chain if you wish
                    self._event_ch.send_nowait(ev)

                    # If a final transcript has text, we can either immediately
                    # return it, or accumulate. Here, we'll just return on the
                    # first non-empty final. That ensures *this* consumption
                    # finishes quickly.
                    if ev.type in SpeechEventType.FINAL_TRANSCRIPT:
                        if ev.alternatives and ev.alternatives[0].text.strip():
                            return ev
            # If the stream ended normally but no final transcript with text
            return None
        except asyncio.TimeoutError:
            logger.warning(
                f"{stt.label} timed out in streaming, switching to next STT",
                extra={"streamed": True},
            )
            raise
        except APIError as e:
            if isinstance(e, APIStatusError) and "Stream timed out" in str(e):
                logger.warning(
                    f"{stt.label} stream timed out, will retry",
                    extra={"streamed": True},
                )
                return None
            logger.warning(
                f"{stt.label} failed in streaming, switching to next STT",
                exc_info=e,
                extra={"streamed": True},
            )
            raise
        except Exception:
            logger.exception(
                f"{stt.label} unexpected error in streaming, switching to next STT",
                extra={"streamed": True},
            )
            raise

    def _try_recovery(self, stt: STT) -> None:
        """
        Launch a "recovery" streaming task in the background. If it sees
        a final transcript, we mark STT as recovered.
        """
        stt_status = self._fallback_adapter._status[
            self._fallback_adapter._stt_instances.index(stt)
        ]
        if (
            stt_status.recovering_stream_task is None
            or stt_status.recovering_stream_task.done()
        ):
            # We open a zero-retry stream with the same adapter
            stream = stt.stream(
                language=self._language,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=0,
                    timeout=self._fallback_adapter._attempt_timeout,
                ),
            )
            self._recovering_streams.append(stream)

            async def _recover_stt_task() -> None:
                try:
                    nb_transcript = 0
                    async with stream:
                        async for ev in stream:
                            if ev.type in SpeechEventType.FINAL_TRANSCRIPT:
                                if ev.alternatives and ev.alternatives[0].text.strip():
                                    nb_transcript += 1
                                    break

                    if nb_transcript == 0:
                        return

                    # If we get at least one final transcript with text, we call that "recovered"
                    stt_status.available = True
                    logger.info(f"tts.FallbackAdapter, {stt.label} recovered")
                    self._fallback_adapter.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"{stream._stt.label} recovery timed out",
                        extra={"streamed": True},
                    )
                except APIError as e:
                    logger.warning(
                        f"{stream._stt.label} recovery failed",
                        exc_info=e,
                        extra={"streamed": True},
                    )
                except Exception:
                    logger.exception(
                        f"{stream._stt.label} recovery unexpected error",
                        extra={"streamed": True},
                    )

            stt_status.recovering_stream_task = task = asyncio.create_task(
                _recover_stt_task()
            )

            # Remove from _recovering_streams when done
            def _remove_on_done(_):
                if stream in self._recovering_streams:
                    self._recovering_streams.remove(stream)

            task.add_done_callback(_remove_on_done)
