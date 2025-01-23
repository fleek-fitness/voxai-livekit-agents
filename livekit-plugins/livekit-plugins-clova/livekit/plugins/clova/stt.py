from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import weakref
from dataclasses import dataclass
from typing import List, Optional, Tuple

import grpc
from grpc import aio

from livekit import rtc
from livekit.agents import (
    stt,
    APIConnectionError,
    APIStatusError,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
)

from . import nest_pb2
from . import nest_pb2_grpc

logger = logging.getLogger(__name__)

# Change port from 443 to 50051 as per Clova docs
CLOVA_SERVER_URL = "clovaspeech-gw.ncloud.com:50051"

DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()

# Audio format requirements from Clova docs
CLOVA_SAMPLE_RATE = 16000  # 16kHz required
CLOVA_BITS_PER_SAMPLE = 16  # 16 bits per sample required
CLOVA_CHANNELS = 1  # 1 channel required
CLOVA_CHUNK_SIZE = 32000  # From Clova example


@dataclass
class ClovaSTTConfig:
    """Configuration for Clova STT following their docs."""

    language: str = "ko"
    keyword_boosting: Optional[List[dict]] = (
        None  # [{"words": "word1,word2", "weight": 1.0}]
    )
    forbidden_words: Optional[List[str]] = None  # ["word1", "word2"]
    semantic_epd: Optional[dict] = (
        None  # {"skipEmptyText": bool, "useWordEpd": bool, ...}
    )
    sample_rate: int = CLOVA_SAMPLE_RATE
    bits_per_sample: int = CLOVA_BITS_PER_SAMPLE
    channels: int = CLOVA_CHANNELS


class STT(stt.STT):
    """
    CLOVA streaming STT plugin that matches the Clova docs:
    https://api.ncloud-docs.com/docs/en/clova-speech-stt-stt
    """

    def __init__(
        self,
        *,
        client_secret: str,
        config: Optional[ClovaSTTConfig] = None,
        default_timeout: float = 120.0,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )

        if not client_secret:
            raise ValueError("Must provide a Clova client_secret token.")

        self._client_secret = client_secret
        self._config = config or ClovaSTTConfig()
        # Change to aio.Channel
        self._channel: aio.Channel | None = None
        self._stub: nest_pb2_grpc.NestServiceStub | None = None
        self._streams = weakref.WeakSet()
        self._default_timeout = default_timeout

        # Use async channel with longer timeouts
        self._channel = aio.secure_channel(
            CLOVA_SERVER_URL,
            credentials=grpc.ssl_channel_credentials(),
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 20000),  # Increased from 10000
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.enable_retries", 1),  # Enable retries
                ("grpc.max_receive_message_length", 10 * 1024 * 1024),  # 10MB
            ],
        )
        # Create stub
        self._stub = nest_pb2_grpc.NestServiceStub(self._channel)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: Optional[str] = None,
    ) -> "ClovaSpeechStream":
        """Create a new streaming session."""
        cfg = dataclasses.replace(self._config)
        if language is not None:
            cfg.language = language

        stream = ClovaSpeechStream(
            stt=self,
            config=cfg,
            conn_options=conn_options,
            client_secret=self._client_secret,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: str | None = None,
    ):
        """Update default config for future streams."""
        if language is not None:
            self._config.language = language

        # Also update active streams
        for s in self._streams:
            s.update_options(language=language)

    async def _recognize_impl(
        self,
        audio_data: bytes,
        language: str | None = None,
    ):
        """
        If you want to do non-streaming usage, implement here.
        For now, we only do streaming => NotImplemented.
        """
        raise NotImplementedError("CLOVA plugin supports streaming only.")

    async def close(self):
        # Clean up streams
        for s in list(self._streams):
            await s.close()
        if self._channel:
            await self._channel.close()
            self._channel = None
        await super().close()


class ClovaSpeechStream(stt.SpeechStream):
    """
    A streaming session to Clova's STT.
    Reads frames from self._input_ch, sends to Clova's gRPC `recognize`, and processes results.
    """

    def __init__(
        self,
        *,
        stt: STT,
        config: ClovaSTTConfig,
        conn_options: APIConnectOptions,
        client_secret: str,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._config = config
        self._client_secret = client_secret
        self._closed = False
        self._buffer = bytearray()
        self._seq_id = 0
        # Add accumulated text buffer
        self._current_text = ""
        self._speech_start_time = None

    def update_options(
        self,
        *,
        language: Optional[str] = None,
    ):
        """Change config in a live stream (if needed)."""
        if language is not None:
            self._config.language = language

    async def close(self) -> None:
        """Close the stream and cleanup resources."""
        if not self._closed:
            self._closed = True
            try:
                if self._input_ch:
                    self._input_ch.close()  # Remove await since Chan.close() is not async
                await asyncio.sleep(0)  # let tasks drain
            finally:
                await super().close()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Ensure the frame matches Clova's requirements:
        - 16kHz sample rate
        - 1 channel (mono)
        - 16-bit samples (int16)
        """
        if self._closed:
            return

        # Convert if audio format doesn't match Clova requirements
        if (
            frame.sample_rate != CLOVA_SAMPLE_RATE
            or frame.num_channels != CLOVA_CHANNELS
        ):
            # Use remix_and_resample for sample rate and channel conversion
            frame = frame.remix_and_resample(
                sample_rate=CLOVA_SAMPLE_RATE, num_channels=CLOVA_CHANNELS
            )

        # Note: AudioFrame already uses 16-bit samples (int16) internally
        # so we don't need to convert bit depth

        super().push_frame(frame)

    async def _run(self) -> None:
        """
        Actual loop that streams audio to Clova's gRPC and yields partial/final results.
        """

        async def request_iterator():
            try:
                # 1) Send CONFIG request with a JSON config
                config_dict = {"transcription": {"language": self._config.language}}

                yield nest_pb2.NestRequest(
                    type=nest_pb2.RequestType.CONFIG,
                    config=nest_pb2.NestConfig(config=json.dumps(config_dict)),
                )

                # 2) Then read frames from the queue and yield DATA requests
                seq_id = 0
                last_activity = time.time()
                while not self._closed:
                    try:
                        # Use async receive with timeout
                        frame = await asyncio.wait_for(
                            self._input_ch.recv(),
                            timeout=30.0,  # 30 second timeout for receiving frames
                        )
                        last_activity = time.time()

                        if frame is None:
                            break

                        if not isinstance(frame, rtc.AudioFrame):
                            continue

                        yield nest_pb2.NestRequest(
                            type=nest_pb2.RequestType.DATA,
                            data=nest_pb2.NestData(
                                chunk=frame.data.tobytes(),
                                extra_contents=json.dumps(
                                    {"seqId": seq_id, "epFlag": False}
                                ),
                            ),
                        )
                        seq_id += 1

                        # Check for inactivity timeout
                        if (
                            time.time() - last_activity > 60
                        ):  # 60 second inactivity timeout
                            logger.warning("Inactivity timeout reached, closing stream")
                            break

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for audio frame")
                        break
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        break

                # Send final chunk with epFlag=True
                if seq_id > 0:
                    yield nest_pb2.NestRequest(
                        type=nest_pb2.RequestType.DATA,
                        data=nest_pb2.NestData(
                            chunk=b"",
                            extra_contents=json.dumps(
                                {"seqId": seq_id + 1, "epFlag": True}
                            ),
                        ),
                    )
            except Exception as e:
                logger.error(f"Error in request iterator: {e}")
                if not self._closed:
                    self._closed = True
                    if self._input_ch:
                        self._input_ch.close()
                raise

        metadata = (("authorization", f"Bearer {self._client_secret}"),)

        stub = self._stt._stub if self._stt else None
        if not stub:
            raise APIConnectionError("Clova stub not initialized")

        try:
            # Use a longer timeout for the gRPC call
            response_stream = stub.recognize(
                request_iterator(),
                metadata=metadata,
                timeout=300.0,  # 5 minute timeout for the entire stream
            )

            # Use async for with response stream
            async for resp in response_stream:
                raw_contents = resp.contents
                try:
                    j = json.loads(raw_contents)

                    # Handle config response
                    if "config" in j:
                        logger.debug(f"Received config response: {j}")
                        continue

                    # Handle transcription response
                    if "transcription" in j:
                        trans_obj = j["transcription"]
                        text = trans_obj.get("text", "")
                        ep_flag = bool(trans_obj.get("epFlag", False))
                        is_final = ep_flag

                        if text:
                            # Track speech start time
                            if self._speech_start_time is None:
                                self._speech_start_time = time.time()
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.START_OF_SPEECH
                                    )
                                )

                            # Accumulate text for interim results
                            if not is_final:
                                self._current_text += text
                                speech_data = stt.SpeechData(
                                    text=self._current_text,  # Send full accumulated text
                                    language=self._config.language,
                                    confidence=1.0,
                                    start_time=self._speech_start_time,
                                    end_time=time.time(),
                                )
                                event = stt.SpeechEvent(
                                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[speech_data],
                                )
                                self._event_ch.send_nowait(event)
                            else:
                                # For final results, send the complete text and reset
                                speech_data = stt.SpeechData(
                                    text=self._current_text + text,
                                    language=self._config.language,
                                    confidence=1.0,
                                    start_time=self._speech_start_time,
                                    end_time=time.time(),
                                )
                                event = stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[speech_data],
                                )
                                self._event_ch.send_nowait(event)

                                # Send end of speech event
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.END_OF_SPEECH
                                    )
                                )

                                # Reset for next utterance
                                self._current_text = ""
                                self._speech_start_time = None

                except ValueError as e:
                    logger.error(f"Failed to parse response: {e}")

        except grpc.RpcError as e:
            code = e.code()
            # Map gRPC codes to your own error classes
            if code == grpc.StatusCode.UNAUTHENTICATED:
                raise APIStatusError("Authentication failed", status_code=401) from e
            elif code == grpc.StatusCode.INVALID_ARGUMENT:
                raise APIStatusError("Invalid parameters", status_code=400) from e
            elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise APIStatusError("Resource exhausted", status_code=429) from e
            elif code == grpc.StatusCode.INTERNAL:
                raise APIStatusError("Internal server error", status_code=500) from e
            else:
                raise APIConnectionError(f"Clova gRPC error: {e}") from e
        except Exception as ex:
            raise APIConnectionError(f"Unexpected error: {ex}") from ex
