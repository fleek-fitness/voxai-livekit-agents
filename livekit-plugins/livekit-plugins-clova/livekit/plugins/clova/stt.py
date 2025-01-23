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

CLOVA_SERVER_URL = "clovaspeech-gw.ncloud.com:443"

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
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )

        if not client_secret:
            raise ValueError("Must provide a Clova client_secret token.")

        self._client_secret = client_secret
        self._config = config or ClovaSTTConfig()
        self._channel: aio.Channel | None = None
        self._stub: nest_pb2_grpc.NestServiceStub | None = None
        self._streams = weakref.WeakSet()

        # Create a gRPC channel once
        self._channel = aio.secure_channel(
            CLOVA_SERVER_URL,
            credentials=grpc.ssl_channel_credentials(),
        )
        # Create the stub
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
            if self._input_ch:
                await self._input_ch.aclose()  # Use aclose() instead of close()

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
                config_dict = {
                    "transcription": {"language": self._config.language},
                    **(
                        {
                            "keywordBoosting": {
                                "boostings": self._config.keyword_boosting
                            }
                        }
                        if self._config.keyword_boosting
                        else {}
                    ),
                    **(
                        {"forbidden": {"forbiddens": self._config.forbidden_words}}
                        if self._config.forbidden_words
                        else {}
                    ),
                    **(
                        {"semanticEpd": self._config.semantic_epd}
                        if self._config.semantic_epd
                        else {}
                    ),
                }

                yield nest_pb2.NestRequest(
                    type=nest_pb2.RequestType.CONFIG,
                    config=nest_pb2.NestConfig(config=json.dumps(config_dict)),
                )

                # 2) Then read frames from the queue and yield DATA requests
                seq_id = 0
                while not self._closed:
                    try:
                        # Use arecv() for async channel receive
                        frame = await self._input_ch.arecv()
                        if frame is None:
                            break

                        if not isinstance(frame, rtc.AudioFrame):
                            continue

                        seq_id += 1
                        yield nest_pb2.NestRequest(
                            type=nest_pb2.RequestType.DATA,
                            data=nest_pb2.NestData(
                                chunk=frame.data.tobytes(),
                                extra_contents=json.dumps(
                                    {"seqId": seq_id, "epFlag": False}
                                ),
                            ),
                        )
                    except asyncio.CancelledError:
                        logger.debug("Stream cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        break

                # Send final chunk with epFlag=True if we have data
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
            finally:
                # Ensure cleanup happens
                if not self._closed:
                    await self.close()

        metadata = (("authorization", f"Bearer {self._client_secret}"),)

        # 3) Start the gRPC streaming call
        stub = self._stt._stub if self._stt else None
        if not stub:
            raise APIConnectionError("Clova stub not initialized")

        try:
            # We pass our request_iterator
            # We also pass a timeout from conn_options
            response_stream = stub.recognize(
                request_iterator(),
                metadata=metadata,
                timeout=self._conn_options.timeout,
            )

            # 4) Process responses from Clova
            async for resp in response_stream:
                raw_contents = resp.contents

                # Clova often returns JSON in resp.contents, but sometimes it can be raw text
                # Attempt to parse as JSON
                is_final = True
                text = ""
                try:
                    j = json.loads(raw_contents)
                    # If there's a top-level key "transcription", parse it
                    if "transcription" in j:
                        trans_obj = j["transcription"]
                        text = trans_obj.get("text", "")
                        # epFlag => if True, treat as final
                        # if False, treat as partial (if you want partial).
                        # We interpret epFlag's presence:
                        ep_flag = bool(trans_obj.get("epFlag", False))
                        is_final = ep_flag  # epFlag=True => final
                    else:
                        # If there's no "transcription", fallback to raw text
                        text = raw_contents
                except ValueError:
                    # Not JSON => treat as raw text
                    text = raw_contents

                # 5) Send a SpeechEvent with either partial or final type
                if text:
                    event_type = (
                        stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    )
                    # Confidence might also come from j["transcription"].get("confidence")
                    # startTimestamp, endTimestamp, etc. can be found if needed
                    speech_data = stt.SpeechData(
                        text=text,
                        language=self._config.language,
                        confidence=1.0,
                        start_time=0,
                        end_time=0,
                    )
                    event = stt.SpeechEvent(
                        type=event_type,
                        alternatives=[speech_data],
                    )
                    await self._event_ch.put(event)

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
