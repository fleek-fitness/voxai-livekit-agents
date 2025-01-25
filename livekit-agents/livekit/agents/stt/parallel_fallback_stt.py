from __future__ import annotations
import asyncio
import time
from ..stt import STT, STTCapabilities, SpeechEvent, SpeechEventType, RecognizeStream
from typing import Optional, Union, AsyncIterator
from livekit import rtc
from ..log import logger
import dataclasses
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio
from livekit.agents._exceptions import APIConnectionError
from ..types import AudioBuffer  # Make sure this import matches your project structure


class ParallelFallbackSTT(STT):
    """
    Parallel STT implementation with fallback logic that:
    1. Uses primary STT's results by default
    2. Falls back to secondary STT only if primary fails to provide interim
    3. Handles both streaming and non-streaming recognition
    """

    def __init__(self, primary: STT, secondary: STT, final_timeout: float = 5.0):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=primary.capabilities.streaming
                and secondary.capabilities.streaming,
                interim_results=primary.capabilities.interim_results,
            )
        )
        self._primary = primary
        self._secondary = secondary
        self._final_timeout = final_timeout

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        """
        Non-streaming recognition with fallback logic
        """
        start_time = time.monotonic()
        timeout_remaining = conn_options.timeout

        try:
            # Try primary first with reduced retries
            return await asyncio.wait_for(
                self._primary.recognize(
                    buffer,
                    language=language,
                    conn_options=dataclasses.replace(
                        conn_options, max_retry=max(0, conn_options.max_retry - 1)
                    ),
                ),
                timeout=timeout_remaining,
            )
        except Exception as primary_err:
            logger.warning(f"Primary STT failed, trying secondary: {str(primary_err)}")

            # Calculate remaining time for secondary attempt
            elapsed = time.monotonic() - start_time
            timeout_remaining = max(0, conn_options.timeout - elapsed)

            try:
                return await asyncio.wait_for(
                    self._secondary.recognize(
                        buffer, language=language, conn_options=conn_options
                    ),
                    timeout=timeout_remaining,
                )
            except Exception as secondary_err:
                raise APIConnectionError(
                    f"Both STTs failed. Primary: {primary_err}, Secondary: {secondary_err}"
                ) from secondary_err

    async def aclose(self) -> None:
        await asyncio.gather(
            self._primary.aclose(), self._secondary.aclose(), return_exceptions=True
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ParallelFallbackStream":
        return ParallelFallbackStream(
            primary=self._primary.stream(language=language, conn_options=conn_options),
            secondary=self._secondary.stream(
                language=language, conn_options=conn_options
            ),
            final_timeout=self._final_timeout,
        )


class ParallelFallbackStream(RecognizeStream):
    def __init__(
        self, primary: RecognizeStream, secondary: RecognizeStream, final_timeout: float
    ):
        super().__init__(
            stt=primary._stt, conn_options=primary._conn_options, sample_rate=None
        )
        self._primary = primary
        self._secondary = secondary
        self._final_timeout = final_timeout
        self._event_ch = aio.Chan[SpeechEvent]()
        self._lock = asyncio.Lock()

        # State management
        self._primary_has_interim = False
        self._accepted_final = False
        self._secondary_final_buffer = None
        self._primary_interim_time = None
        self._pending_timers = set()

        # Start processing tasks
        self._primary_task = asyncio.create_task(self._process_primary())
        self._secondary_task = asyncio.create_task(self._process_secondary())
        self._merged_aiter = self._merge_events()

    async def _process_primary(self):
        try:
            async for ev in self._primary:
                async with self._lock:
                    if self._accepted_final:
                        return

                    if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                        self._primary_has_interim = True
                        self._primary_interim_time = time.monotonic()
                        self._event_ch.send_nowait(ev)
                        self._secondary_final_buffer = None
                        self._start_primary_timeout()

                    elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        self._accepted_final = True
                        self._event_ch.send_nowait(ev)
                        await self._close_streams()
        finally:
            await self._primary.aclose()

    async def _process_secondary(self):
        try:
            async for ev in self._secondary:
                async with self._lock:
                    if self._accepted_final:
                        return

                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        if not self._primary_has_interim:
                            # Immediate acceptance if no primary interim
                            self._accepted_final = True
                            self._event_ch.send_nowait(ev)
                            await self._close_streams()
                        else:
                            # Store for potential timeout fallback
                            self._secondary_final_buffer = ev
        finally:
            await self._secondary.aclose()

    def _start_primary_timeout(self):
        async def _timeout_check():
            await asyncio.sleep(self._final_timeout)
            async with self._lock:
                if not self._accepted_final and self._secondary_final_buffer:
                    self._accepted_final = True
                    self._event_ch.send_nowait(self._secondary_final_buffer)
                    await self._close_streams()

        timer = asyncio.create_task(_timeout_check())
        self._pending_timers.add(timer)
        timer.add_done_callback(lambda t: self._pending_timers.discard(t))

    async def _merge_events(self) -> AsyncIterator[SpeechEvent]:
        try:
            async for ev in self._event_ch:
                yield ev
        finally:
            self._event_ch.close()

    async def _close_streams(self):
        """Handle stream closure and pending timeouts"""
        for timer in self._pending_timers:
            timer.cancel()

        await asyncio.gather(
            self._primary.aclose(), self._secondary.aclose(), return_exceptions=True
        )

        if not self._accepted_final and self._secondary_final_buffer:
            self._event_ch.send_nowait(self._secondary_final_buffer)

    # Required stream interface methods
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._primary.push_frame(frame)
        self._secondary.push_frame(frame)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def end_input(self) -> None:
        self._primary.end_input()
        self._secondary.end_input()

    async def aclose(self) -> None:
        await self._close_streams()
        await asyncio.gather(
            self._primary_task, self._secondary_task, return_exceptions=True
        )
        self._event_ch.close()

    async def __anext__(self) -> SpeechEvent:
        try:
            return await self._merged_aiter.__anext__()
        except StopAsyncIteration:
            for task in [self._primary_task, self._secondary_task]:
                if task.done() and not task.cancelled() and task.exception():
                    raise task.exception()
            raise

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self
