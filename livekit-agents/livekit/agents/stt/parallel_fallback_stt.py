from __future__ import annotations
import asyncio
import time
import dataclasses
from typing import Optional, Union, AsyncIterator
from livekit import rtc
from ..log import logger
from .stt import STT, STTCapabilities, SpeechEvent, SpeechEventType, RecognizeStream
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio
from livekit.agents._exceptions import APIConnectionError
from livekit.agents.utils.audio import AudioBuffer


class ParallelFallbackSTT(STT):
    """
    Parallel STT implementation with fallback logic that:
    1. Uses primary STT's results by default
    2. Falls back to secondary STT only if primary fails to provide interim
    3. Handles both streaming and non-streaming recognition
    """

    def __init__(
        self,
        primary: STT,
        secondary: STT,
        final_timeout: float = 5.0,
        idle_timeout: float = 5.0,
    ):
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
        self._idle_timeout = idle_timeout  # Add idle timeout

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
            idle_timeout=self._idle_timeout,  # Pass idle_timeout to stream
        )


class ParallelFallbackStream(RecognizeStream):
    IDLE_TIMEOUT = 3.0  # Define idle timeout as a class constant
    FINAL_COOLDOWN_PERIOD = 1.0  # Add final cooldown period

    def __init__(
        self,
        primary: RecognizeStream,
        secondary: RecognizeStream,
        final_timeout: float,
        idle_timeout: float,
    ):
        super().__init__(
            stt=primary._stt, conn_options=primary._conn_options, sample_rate=None
        )
        self._primary = primary
        self._secondary = secondary
        self._final_timeout = final_timeout
        self._idle_timeout = idle_timeout  # Use passed idle_timeout
        self._event_ch = aio.Chan[SpeechEvent]()
        self._lock = asyncio.Lock()  # Instance lock for _check_final
        self._should_restart = asyncio.Event()

        # Add stream state tracking
        self._primary_active = asyncio.Event()
        self._secondary_active = asyncio.Event()
        self._primary_active.set()
        self._secondary_active.set()

        # State management for finals and activity
        self._candidate_primary_final: Optional[SpeechEvent] = None
        self._candidate_secondary_final: Optional[SpeechEvent] = None
        self._accepted_final = False
        self._last_primary_activity = 0.0  # Track primary activity for idle timeout
        self._last_final_accepted_time = (
            0.0  # Track last final accepted time for cooldown
        )

        # Start processing tasks
        self._primary_task = asyncio.create_task(self._process_primary())
        self._secondary_task = asyncio.create_task(self._process_secondary())
        self._merged_aiter = self._merge_events()
        self._idle_task = asyncio.create_task(self._idle_loop())  # Start idle loop

    def _reset_state(self) -> None:
        """Reset internal state for next turn"""
        self._candidate_primary_final = None
        self._candidate_secondary_final = None
        self._accepted_final = False
        self._last_primary_activity = 0.0
        # self._last_final_accepted_time = 0.0 # Reset final accepted time

    async def _idle_loop(self):
        """Periodically check for primary inactivity and trigger fallback if needed."""
        try:
            while not self._accepted_final:
                await asyncio.sleep(0.1)  # Check every 0.1 second
                await self._check_idle_time()
        except asyncio.CancelledError:
            pass  # Expected on stream close

    async def _check_idle_time(self):
        """If primary is silent for too long, consider fallback to secondary."""
        async with self._lock:  # Use lock here as well to be consistent and prevent race in idle timeout check.
            if self._accepted_final and (
                time.monotonic() - self._last_final_accepted_time
                < self.FINAL_COOLDOWN_PERIOD
            ):
                return  # In cooldown period

            if self._accepted_final:  # Already accepted a final
                return

            now = time.monotonic()
            if (
                now - self._last_primary_activity
            ) > self._idle_timeout and self._candidate_secondary_final:
                logger.info(
                    "Idle timeout reached, falling back to secondary STT final."
                )
                await self._accept_final(self._candidate_secondary_final)
            elif (
                (now - self._last_primary_activity) > self._final_timeout
                and not self._candidate_primary_final
                and self._candidate_secondary_final
            ):  # Fallback after final timeout if primary failed to give final
                logger.warning(
                    "Final timeout reached and primary STT has no final, falling back to secondary STT final."
                )
                await self._accept_final(self._candidate_secondary_final)
            # elif (now - self._last_primary_activity) > self._final_timeout and not self._candidate_primary_final and not self._candidate_secondary_final: # Fallback to empty final after final timeout if neither STT has final
            #     logger.warning("Final timeout reached and neither STT has final, sending empty final.")
            #     empty_final_event = SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[rtc.SpeechAlternative(text="", confidence=1.0)])
            #     await self._accept_final(empty_final_event)

    async def _accept_final(self, final_event: SpeechEvent):
        """Centralized method to accept and emit a final event."""
        if (
            self._accepted_final
        ):  # Prevent emitting multiple finals - Double check here as well.
            return
        if (
            time.monotonic() - self._last_final_accepted_time
            < self.FINAL_COOLDOWN_PERIOD
        ):
            logger.info(
                f"Final cooldown period not over, skipping emission. Last final accepted time: {self._last_final_accepted_time}, current time: {time.monotonic()}"
            )
            self._reset_state()
            return
        self._accepted_final = True
        self._last_final_accepted_time = time.monotonic()  # Record final accepted time
        logger.info(
            f"Emitting final transcript: {final_event.alternatives[0].text if final_event.alternatives else ''} from STT: {final_event.stt_name if hasattr(final_event, 'stt_name') else 'Unknown'}"
        )  # Log source of final
        self._event_ch.send_nowait(final_event)
        self._should_restart.set()

    async def _check_final(self):
        """Determine which final to accept (primary, secondary, or best)."""
        async with self._lock:  # Acquire lock to make _check_final atomic
            if self._accepted_final and (
                time.monotonic() - self._last_final_accepted_time
                < self.FINAL_COOLDOWN_PERIOD
            ):
                return  # In cooldown period

            if self._accepted_final:
                return

            if self._candidate_primary_final:
                logger.info("Primary final available, using primary STT final.")
                await self._accept_final(self._candidate_primary_final)
                return
            elif self._candidate_secondary_final:
                # Check idle time first before accepting secondary immediately in _check_idle_time
                now = time.monotonic()
                if (now - self._last_primary_activity) > (
                    self._idle_timeout / 2.0
                ):  # Shorter grace period before considering secondary final immediately if available
                    logger.info(
                        "Secondary final available and primary idle, using secondary STT final."
                    )
                    await self._accept_final(self._candidate_secondary_final)
                    return
                else:
                    logger.info(
                        "Secondary final available, waiting for primary or idle timeout."
                    )
                    return

    async def _process_primary(self):
        try:
            async for ev in self._primary:
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    self._last_primary_activity = time.monotonic()
                    logger.info(f"STT-1 INTERIM: {ev.alternatives[0].text}")
                    ev.stt_name = "STT-1"  # Add STT name for logging
                    self._event_ch.send_nowait(ev)

                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    logger.info(f"STT-1 FINAL: {ev.alternatives[0].text}")
                    ev.stt_name = "STT-1"  # Add STT name for logging
                    async with self._lock:  # Acquire lock before processing final
                        if self._accepted_final:  # Re-check inside lock!
                            continue  # Already accepted a final, discard this one
                        self._candidate_primary_final = ev  # Store primary final
                    await self._check_final()  # Centralized final check

        except Exception as e:
            logger.error("Primary STT stream failed", exc_info=e)
        finally:
            self._primary_active.clear()

    async def _process_secondary(self):
        try:
            async for ev in self._secondary:
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    logger.info(f"STT-2 INTERIM: {ev.alternatives[0].text}")
                    ev.stt_name = "STT-2"  # Add STT name for logging

                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    logger.info(f"STT-2 FINAL: {ev.alternatives[0].text}")
                    ev.stt_name = "STT-2"  # Add STT name for logging
                    async with self._lock:  # Acquire lock before processing final
                        if self._accepted_final:  # Re-check inside lock!
                            continue  # Already accepted a final, discard this one
                        self._candidate_secondary_final = ev  # Store secondary final
                    await self._check_final()  # Centralized final check

        except Exception as e:
            logger.error("Secondary STT stream failed", exc_info=e)
        finally:
            self._secondary_active.clear()

    async def _merge_events(self) -> AsyncIterator[SpeechEvent]:
        try:
            async for ev in self._event_ch:
                yield ev
        finally:
            self._event_ch.close()

    async def _close_streams(self):
        """Handle stream closure and pending timeouts"""
        if not self._accepted_final and self._candidate_secondary_final:
            await self._accept_final(self._candidate_secondary_final)

        await asyncio.gather(
            self._primary.aclose(), self._secondary.aclose(), return_exceptions=True
        )

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Handle frame pushing with active stream checks"""
        if self._primary_active.is_set():
            try:
                self._primary.push_frame(frame)
            except RuntimeError as e:
                if "input ended" in str(e):
                    logger.warning("Primary stream closed, stopping frame pushes")
                    self._primary_active.clear()
                else:
                    raise

        if self._secondary_active.is_set():
            try:
                self._secondary.push_frame(frame)
            except RuntimeError as e:
                if "input ended" in str(e):
                    logger.warning("Secondary stream closed, stopping frame pushes")
                    self._secondary_active.clear()
                else:
                    raise

        if not self._primary_active.is_set() and not self._secondary_active.is_set():
            raise APIConnectionError("All STT streams have failed")

    def flush(self) -> None:
        if self._primary_active.is_set():
            self._primary.flush()
        if self._secondary_active.is_set():
            self._secondary.flush()

    def end_input(self) -> None:
        if self._primary_active.is_set():
            self._primary.end_input()
        if self._secondary_active.is_set():
            self._primary.end_input()
        if self._secondary_active.is_set():
            self._secondary.end_input()

    async def aclose(self) -> None:
        """Handle cleanup with proper cancellation"""
        self._idle_task.cancel()  # Cancel idle loop
        tasks = [
            self._primary_task,
            self._secondary_task,
            self._idle_task,
        ]  # Include idle task
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        await self._close_streams()  # Call close streams to handle pending final
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

    async def _run(self) -> None:
        """Main processing loop required by RecognizeStream"""
        try:
            while True:
                await self._should_restart.wait()
                async with self._lock:
                    self._reset_state()
                    self._should_restart.clear()
                    # Reactivate both streams for next utterance
                    self._primary_active.set()
                    self._secondary_active.set()
        except Exception as e:
            logger.error(f"ParallelFallbackStream failed: {str(e)}")
            raise
