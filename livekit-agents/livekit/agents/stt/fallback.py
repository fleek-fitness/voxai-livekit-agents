from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from livekit import rtc

from . import STT, STTCapabilities, SpeechEvent, StreamAdapter
from ..log import logger


@dataclass
class StoredResult:
    """Stores STT result"""

    event: SpeechEvent
    provider: STT


class FallbackSTT(StreamAdapter):
    """
    A wrapper that runs two STT providers in parallel, using whichever gives a valid result first,
    while preferring the primary provider when both return results.
    """

    def __init__(
        self,
        primary: STT,
        secondary: STT,
    ):
        # Calculate capabilities before calling super().__init__()
        capabilities = STTCapabilities(
            streaming=(primary.capabilities.streaming),
            interim_results=(primary.capabilities.interim_results),
        )

        # Pass capabilities to parent class
        super().__init__(capabilities=capabilities)

        self._primary = primary
        self._secondary = secondary

        # Store results from both providers
        self._primary_result: Optional[StoredResult] = None
        self._secondary_result: Optional[StoredResult] = None
        self._result_emitted = False

        self._wire_event_listeners()

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    def _wire_event_listeners(self):
        """Set up event listeners for both STT providers."""

        @self._primary.on("start_of_speech")
        @self._secondary.on("start_of_speech")
        def _on_start_of_speech(ev: Any):
            # Reset state for new speech segment
            self._primary_result = None
            self._secondary_result = None
            self._result_emitted = False
            self.emit("start_of_speech", ev)

        @self._primary.on("end_of_speech")
        @self._secondary.on("end_of_speech")
        def _on_end_of_speech(ev: Any):
            self.emit("end_of_speech", ev)

        @self._primary.on("interim_transcript")
        def _on_primary_interim(ev: SpeechEvent):
            # Always emit primary interim if we haven't emitted final
            if not self._result_emitted:
                self.emit("interim_transcript", ev)

        @self._secondary.on("interim_transcript")
        def _on_secondary_interim(ev: SpeechEvent):
            # Only emit secondary interim if primary hasn't produced anything yet
            if not self._result_emitted and not self._primary_result:
                self.emit("interim_transcript", ev)

        @self._primary.on("final_transcript")
        def _on_primary_final(ev: SpeechEvent):
            if self._result_emitted:
                return

            text = ev.alternatives[0].text.strip()
            if text:
                self._primary_result = StoredResult(event=ev, provider=self._primary)
                # Primary has content - use it immediately
                logger.debug("Using primary STT result", extra={"text": text})
                self.emit("final_transcript", ev)
                self._result_emitted = True
            elif self._secondary_result:
                # Primary is empty but we have secondary - use secondary
                logger.debug(
                    "Using secondary STT result (primary empty)",
                    extra={"text": self._secondary_result.event.alternatives[0].text},
                )
                self.emit("final_transcript", self._secondary_result.event)
                self._result_emitted = True

        @self._secondary.on("final_transcript")
        def _on_secondary_final(ev: SpeechEvent):
            if self._result_emitted:
                return

            text = ev.alternatives[0].text.strip()
            if text:
                self._secondary_result = StoredResult(
                    event=ev, provider=self._secondary
                )

                if self._primary_result:
                    # We have primary result - only use secondary if primary was empty
                    primary_text = self._primary_result.event.alternatives[
                        0
                    ].text.strip()
                    if primary_text:
                        logger.debug(
                            "Using primary STT result (both available)",
                            extra={
                                "primary_text": primary_text,
                                "secondary_text": text,
                            },
                        )
                        self.emit("final_transcript", self._primary_result.event)
                        self._result_emitted = True

    async def start_processing(self, track: rtc.Track):
        """Start processing audio with both providers."""
        await asyncio.gather(
            self._primary.start_processing(track),
            self._secondary.start_processing(track),
        )

    async def stop_processing(self):
        """Stop processing audio with both providers."""
        await asyncio.gather(
            self._primary.stop_processing(), self._secondary.stop_processing()
        )

    async def aclose(self):
        """Close both STT providers."""
        await asyncio.gather(self._primary.aclose(), self._secondary.aclose())
