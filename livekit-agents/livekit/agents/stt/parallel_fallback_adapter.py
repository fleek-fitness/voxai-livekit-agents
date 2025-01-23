import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Literal, List, Optional

from livekit import rtc
from livekit.agents.utils.audio import AudioBuffer

from .._exceptions import APIConnectionError, APIError
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from ..log import logger
from .stt import STT, STTCapabilities, RecognizeStream, SpeechEvent, SpeechEventType

DEFAULT_PARALLEL_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0,
    timeout=DEFAULT_API_CONNECT_OPTIONS.timeout,
)


@dataclass
class FallbackResult:
    """
    Simple helper to store which STT succeeded and the final transcript event.
    """

    stt_index: int
    event: SpeechEvent


class ParallelFallbackSTT(STT[Literal["stt_availability_changed"]]):
    """
    Run multiple STT providers in parallel for each utterance:
      1) Everyone is fed the same audio in real time.
      2) We wait on final transcripts from the STTs in priority order:
         - If STT #0 has a final transcript with non-empty text => done
         - Else if STT #0 was empty or failed => check STT #1, etc.
      3) If none yield a non-empty final, raise an error.
    """

    def __init__(
        self,
        stt_list: List[STT],
        conn_options: APIConnectOptions = DEFAULT_PARALLEL_API_CONNECT_OPTIONS,
    ):
        if not stt_list:
            raise ValueError("You must provide at least one STT instance.")
        self._stt_list = stt_list
        self._conn_options = conn_options

        # Aggregate capabilities: for parallel usage, we typically need them all to be streaming.
        can_stream = all(stt.capabilities.streaming for stt in stt_list)
        can_interim = all(stt.capabilities.interim_results for stt in stt_list)

        super().__init__(
            capabilities=STTCapabilities(
                streaming=can_stream, interim_results=can_interim
            )
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str] = None,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> SpeechEvent:
        """
        Non-streaming fallback: feed the same buffer to each STT in parallel,
        collect results, pick the first non-empty final in priority order.
        """
        if conn_options is None:
            conn_options = self._conn_options

        start_time = time.time()

        # Kick off .recognize() on all STTs in parallel
        tasks = []
        for i, stt in enumerate(self._stt_list):
            t = asyncio.create_task(
                self._recognize_one_stt(i, stt, buffer, language, conn_options)
            )
            tasks.append(t)

        done, _pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # Gather results
        results_map: dict[int, SpeechEvent] = {}
        errors_map: dict[int, Exception] = {}

        for t in done:
            stt_index, result_or_exc = t.result()
            if isinstance(result_or_exc, Exception):
                errors_map[stt_index] = result_or_exc
            else:
                results_map[stt_index] = result_or_exc

        # Evaluate in priority order
        final_event = self._pick_result_in_priority_order(results_map, errors_map)
        if not final_event:
            stt_labels = [stt.label for stt in self._stt_list]
            raise APIConnectionError(
                f"All STTs either failed or produced empty transcript ({stt_labels}) "
                f"after {time.time() - start_time:.1f}s"
            )

        return final_event

    async def _recognize_one_stt(
        self,
        idx: int,
        stt: STT,
        buffer: AudioBuffer,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ):
        """
        Calls stt.recognize() -> returns (index, SpeechEvent or Exception).
        """
        try:
            ev = await stt.recognize(
                buffer, language=language, conn_options=conn_options
            )
            return (idx, ev)
        except Exception as e:
            return (idx, e)

    def stream(
        self,
        *,
        language: Optional[str] = None,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> RecognizeStream:
        """
        Parallel fallback streaming aggregator. We'll feed frames to all STTs at once,
        and pick the first non-empty final in priority order.
        """
        if conn_options is None:
            conn_options = self._conn_options

        return ParallelFallbackRecognizeStream(
            stt_list=self._stt_list,
            parent_stt=self,
            language=language,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        """
        If your pipeline calls aclose() on this STT, forward to each child STT.
        """
        for stt in self._stt_list:
            with contextlib.suppress(Exception):
                await stt.aclose()

    def _pick_result_in_priority_order(
        self,
        results_map: dict[int, SpeechEvent],
        errors_map: dict[int, Exception],
    ) -> Optional[SpeechEvent]:
        """
        Among all STTs that produced a final transcript, pick the first STT (index order)
        that yields a non-empty final transcript.
        """
        for idx in range(len(self._stt_list)):
            if idx in results_map:
                ev = results_map[idx]
                if ev.alternatives and ev.alternatives[0].text.strip():
                    return ev
                # else empty => check next STT
            # if error or no entry => skip
        return None


class ParallelFallbackRecognizeStream(RecognizeStream):
    """
    Streaming aggregator:
      - Creates a sub-stream for each STT in self._stt_list
      - Feeds audio frames to all
      - Collects final transcript events
      - As soon as the primary STT (index 0) yields a non-empty final -> finalize
      - If primary is empty or fails -> see if the next STT yields a non-empty final, etc.
    """

    def __init__(
        self,
        stt_list: List[STT],
        parent_stt: ParallelFallbackSTT,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=parent_stt, conn_options=conn_options, sample_rate=None)
        self._language = language
        self._stt_list = stt_list

        # We'll create sub-streams for each STT in parallel
        self._sub_streams: List[RecognizeStream] = []
        # Final transcripts from each STT
        self._final_events: dict[int, SpeechEvent] = {}
        self._errors: dict[int, Exception] = {}
        # Event to indicate we can finalize
        self._done = asyncio.Event()

    async def _run(self) -> None:
        start_time = time.time()

        # 1) Create sub-streams
        for stt in self._stt_list:
            s = stt.stream(language=self._language, conn_options=self._conn_options)
            self._sub_streams.append(s)

        # 2) Start reading from each sub-stream in parallel
        read_tasks = []
        for idx, s in enumerate(self._sub_streams):
            t = asyncio.create_task(self._read_sub_stream(idx, s))
            read_tasks.append(t)

        # 3) Forward audio frames from self._input_ch to each sub-stream
        forward_task = asyncio.create_task(self._forward_input())

        # 4) Wait for either:
        #    - self._done to be set (meaning we decided on a final result),
        #    - or for all read tasks to complete (meaning no one gave a good final).
        done_set, pending = await asyncio.wait(
            [self._done.wait(), *read_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If self._done is set, we have a final => cancel reading tasks
        if self._done.is_set():
            for t in read_tasks:
                if not t.done():
                    t.cancel()
        else:
            # All read tasks finished but we never finalized => no STT gave a non-empty final
            self._done.set()

        # Cancel forward input
        forward_task.cancel()
        await asyncio.gather(*read_tasks, return_exceptions=True)
        await aio.gracefully_cancel(forward_task)

        # 5) Pick final result in priority order
        final_event = self._pick_result_in_priority_order()
        if not final_event:
            stt_labels = [stt.label for stt in self._stt_list]
            raise APIConnectionError(
                f"All STTs ended but produced empty/fail: {stt_labels}, "
                f"after {time.time() - start_time:.1f}s"
            )

        # 6) Emit the final transcript
        self._event_ch.send_nowait(final_event)

    async def _read_sub_stream(self, idx: int, sub_stream: RecognizeStream) -> None:
        """
        Read events from a single sub-stream, gather a final transcript if any.
        If the primary STT yields a non-empty final, we can finalize immediately.
        """
        try:
            async with sub_stream:
                async for event in sub_stream:
                    # If you want to forward interim from the primary only, you can do:
                    # if idx == 0 and event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    #     self._event_ch.send_nowait(event)

                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        self._final_events[idx] = event
                        # If this is the primary STT (idx=0) and text is non-empty => finalize
                        if (
                            idx == 0
                            and event.alternatives
                            and event.alternatives[0].text.strip()
                        ):
                            self._done.set()
                            return
        except Exception as e:
            self._errors[idx] = e

    async def _forward_input(self) -> None:
        """
        Push audio frames from self._input_ch to all sub-streams.
        """
        try:
            async for data in self._input_ch:
                for s in self._sub_streams:
                    if isinstance(data, rtc.AudioFrame):
                        s.push_frame(data)
                    else:
                        s.flush()
        finally:
            # Signal no more data
            for s in self._sub_streams:
                with contextlib.suppress(Exception):
                    s.end_input()

    def _pick_result_in_priority_order(self) -> Optional[SpeechEvent]:
        """
        Among the final transcripts we have, pick the first non-empty in index order.
        """
        for idx in range(len(self._stt_list)):
            if idx in self._errors:
                continue
            if idx not in self._final_events:
                continue
            event = self._final_events[idx]
            if event.alternatives and event.alternatives[0].text.strip():
                return event
        return None
