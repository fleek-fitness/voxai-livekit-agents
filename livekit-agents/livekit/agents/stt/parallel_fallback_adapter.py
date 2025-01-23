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
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class FallbackResult:
    """
    Just a small helper to store who succeeded, what text was produced, etc.
    """

    stt_index: int
    event: SpeechEvent


class ParallelFallbackSTT(STT[Literal["stt_availability_changed"]]):
    """
    Run multiple STT providers in parallel for each utterance:
      1) Everyone is fed the same audio in real time.
      2) We wait on final transcripts from the STTs **in priority order**:
         - If STT #0 has a final transcript with non-empty text => done
         - Else if STT #0 was empty or failed => check STT #1’s final transcript
         - and so on...
    """

    def __init__(
        self,
        stt_list: List[STT],
        # Optional: pick an STT if it finishes in N seconds, or else we fallback
        # But truly parallel means each STT is capturing from the start, no “timeout switch.”
        # If you want custom timeouts for each STT, you can store them here.
        conn_options: APIConnectOptions = DEFAULT_PARALLEL_API_CONNECT_OPTIONS,
    ):
        if not stt_list:
            raise ValueError("You must provide at least one STT instance.")
        self._stt_list = stt_list
        self._conn_options = conn_options

        # Aggregate capabilities:
        # For “parallel,” we typically require that all STTs are streaming, or
        # you must wrap non-streaming ones in a stream adapter.
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
        conn_options: APIConnectOptions = None,
    ) -> SpeechEvent:
        """
        Non-streaming fallback. We'll feed the same buffer to each STT in parallel,
        gather results/failures, and pick the first non-empty text from the primary-first order.
        """
        if conn_options is None:
            conn_options = self._conn_options

        start_time = time.time()

        # Kick off all STTs in parallel, each one does .recognize()
        tasks = []
        for i, stt in enumerate(self._stt_list):
            t = asyncio.create_task(
                self._recognize_one_stt(i, stt, buffer, language, conn_options)
            )
            tasks.append(t)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # Collect results and exceptions in a dictionary
        results_map = {}  # stt_index -> FallbackResult
        errors_map = {}  # stt_index -> Exception

        for t in done:
            stt_index, result_or_exc = t.result()
            if isinstance(result_or_exc, Exception):
                errors_map[stt_index] = result_or_exc
            else:
                # It's a SpeechEvent
                results_map[stt_index] = result_or_exc

        # Evaluate in priority order
        #   - If the primary STT (index=0) gave a non-empty final => use it
        #   - Otherwise, if it was empty or failed => check STT #1, etc.
        final_event = self._pick_result_in_priority_order(results_map, errors_map)

        if not final_event:
            # All STTs either failed or returned empty transcript
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
        Just calls stt.recognize() and returns either (idx, SpeechEvent) or (idx, Exception).
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
        conn_options: APIConnectOptions = None,
    ) -> RecognizeStream:
        """
        Return a "parallel fallback" streaming aggregator.
        We'll feed frames to all STT streams in parallel, watch final transcripts,
        and pick from them in priority order.
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
        # If your pipeline calls aclose() on the STT, you can propagate
        # to each child STT if necessary.
        for stt in self._stt_list:
            with contextlib.suppress(Exception):
                await stt.aclose()

    def _pick_result_in_priority_order(
        self, results_map: dict[int, SpeechEvent], errors_map: dict[int, Exception]
    ) -> SpeechEvent | None:
        """
        Among all STTs that produced a final transcript event,
        pick the first STT in priority order with a *non-empty* final.
        """
        for idx in range(len(self._stt_list)):
            if idx in results_map:
                ev = results_map[idx]
                # We only need the FIRST final alternative
                if ev.alternatives and ev.alternatives[0].text.strip():
                    return ev
                else:
                    # This STT had an empty final transcript
                    # => fallback to the next STT
                    pass

            # If it was an error, we skip it
            # If no entry in results_map or errors_map => that STT never completed
            # (shouldn't happen because we wait for them to finish in .recognize())

        return None


class ParallelFallbackRecognizeStream(RecognizeStream):
    """
    Streaming aggregator:
      - Creates a sub-stream for each STT in self._stt_list.
      - Feeds audio frames to all.
      - Collects final transcript events. As soon as the primary STT
        yields a non-empty final => we emit it and close everything.
      - If the primary yields an empty final => we move on to the secondary, etc.
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

        # We'll create sub-streams for each child STT:
        self._sub_streams: List[RecognizeStream] = []
        # We track the final transcript event from each STT (if any).
        self._final_events: dict[int, SpeechEvent] = {}
        # Also track if a sub-stream ended with an error
        self._errors: dict[int, Exception] = {}

        # We'll keep a simple event to detect "did we finalize?"
        self._done = asyncio.Event()

    async def _run(self) -> None:
        """
        Core logic of the aggregator:
          1) Create sub-streams for each STT in parallel.
          2) Start reading from them in separate tasks, gather their partial/final events.
          3) As soon as we get a final from the primary that is non-empty => finalize.
             Otherwise, if it’s empty => we proceed to the second STT’s final, etc.
        """
        start_time = time.time()

        # 1) Create all sub-streams
        for i, stt in enumerate(self._stt_list):
            s = stt.stream(language=self._language, conn_options=self._conn_options)
            self._sub_streams.append(s)

        # 2) We'll have tasks that read from each sub-stream.
        read_tasks = []
        for idx, s in enumerate(self._sub_streams):
            t = asyncio.create_task(self._read_sub_stream(idx, s))
            read_tasks.append(t)

        # 3) Also create a forward-input task: all frames we get from self._input_ch
        #    are forwarded to each sub-stream.
        forward_task = asyncio.create_task(self._forward_input())

        # Wait for self._done to be set, or for all sub-streams to end
        # (If all sub-streams ended but we never found a non-empty final => fail)
        done_set, pending = await asyncio.wait(
            [self._done.wait(), *read_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If the event `_done` is set, it means we've decided on a result
        # (or decided there's no result).
        # Cancel all reading tasks and the forward task.
        for t in read_tasks:
            if not t.done():
                t.cancel()
        forward_task.cancel()

        # Let’s see if we are done because `_done` was set by some final or
        # because all read_tasks ended:
        if self._done.is_set():
            # We must flush out the results, i.e. the final pick
            pass
        else:
            # Means all read_tasks completed without setting `_done`
            # => No STT gave us a final transcript, or they all failed/empty.
            self._done.set()

        # Let’s wait for the tasks to truly exit
        await asyncio.gather(*read_tasks, return_exceptions=True)
        await aio.gracefully_cancel(forward_task)

        # 4) We pick the final in priority order, same logic as non-streaming
        final_event = self._pick_result_in_priority_order()
        if not final_event:
            stt_labels = [stt.label for stt in self._stt_list]
            raise APIConnectionError(
                f"All STTs ended but produced empty/fail: {stt_labels}, "
                f"after {time.time() - start_time:.1f}s"
            )

        # 5) Emit the final transcript event outward
        self._event_ch.send_nowait(final_event)

    async def _read_sub_stream(self, idx: int, sub_stream: RecognizeStream) -> None:
        """
        Reads events from a single sub-stream until final or error.
        If we get partial transcripts, we can choose to forward them to self._event_ch
        (so the user sees interim from the chosen STT?).
        But often you'd only forward interim from the primary or
        not forward interim at all until you decide which STT is final.
        """
        try:
            async with sub_stream:
                async for event in sub_stream:
                    # Forward *interim* from whichever STT you want.
                    # Usually you'd only forward from the primary
                    # if idx == 0 and event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    #     self._event_ch.send_nowait(event)

                    if event.type in (SpeechEventType.FINAL_TRANSCRIPT,):
                        # Store final event
                        self._final_events[idx] = event

                        # Now check if we can finalize:
                        # If idx=0 (primary), we finalize immediately if text is non-empty
                        # Else if idx=0 text is empty -> we keep going to see if there's a next final?
                        # Actually we typically see only one final per STT. So let's handle it:

                        # 1) Mark that sub_stream is done reading
                        # 2) Possibly set self._done if the primary STT had text,
                        #    or we suspect the next STT might produce text.
                        #    But we can't finalize *just yet* if idx=0 is empty
                        #    because we might need index=1.

                        # Actually simpler: we just store the final event and let
                        # `_pick_result_in_priority_order` handle it after all sub-streams are done,
                        # or after a short wait. But we can also do "as soon as we see
                        # the primary has text, we finalize."

                        if idx == 0:
                            # Primary STT
                            if (
                                event.alternatives
                                and event.alternatives[0].text.strip()
                            ):
                                # Non-empty => finalize now
                                self._done.set()
                                return
                            else:
                                # It's empty => we do not finalize immediately
                                # We want to see if STT #1 or #2 produce anything better.
                                # So do nothing, keep reading.
                                pass
                        else:
                            # For secondary STTs:
                            # We only finalize if the primary final is known
                            # to be empty/fail or doesn't exist yet.
                            # But let's handle that in `_done` approach for simplicity.
                            pass
        except Exception as e:
            # e.g. APIError, Timeout, etc.
            self._errors[idx] = e
        finally:
            # sub_stream ended. That’s fine.
            pass

    async def _forward_input(self) -> None:
        """
        Reads audio frames from self._input_ch (the main RecognizeStream queue)
        and pushes them to each sub_stream.
        """
        try:
            async for data in self._input_ch:
                for s in self._sub_streams:
                    if isinstance(data, rtc.AudioFrame):
                        s.push_frame(data)
                    else:
                        # flush or other sentinel
                        s.flush()
        finally:
            # End input on all sub-streams
            for s in self._sub_streams:
                with contextlib.suppress(Exception):
                    s.end_input()

    def _pick_result_in_priority_order(self) -> SpeechEvent | None:
        """
        Among the final events we have, pick the first STT (index order) that
        yields a non-empty final transcript.
        """
        for idx in range(len(self._stt_list)):
            # If there was an error, skip
            if idx in self._errors:
                continue
            if idx not in self._final_events:
                continue
            ev = self._final_events[idx]
            if ev.alternatives and ev.alternatives[0].text.strip():
                return ev
        return None
