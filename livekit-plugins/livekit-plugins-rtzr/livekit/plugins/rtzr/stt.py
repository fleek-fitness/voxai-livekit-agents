from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import weakref
from dataclasses import dataclass
from typing import List, Tuple

import grpc
import requests
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)

from . import vito_stt_client_pb2 as pb
from . import vito_stt_client_pb2_grpc as pb_grpc
from .log import logger
from .models import RTZRLanguages, RTZRModels

GRPC_SERVER_URL = "grpc-openapi.vito.ai:443"
API_BASE = "https://openapi.vito.ai"


@dataclass
class STTOptions:
    language: RTZRLanguages | str | None
    model: RTZRModels
    sample_rate: int
    num_channels: int
    use_itn: bool
    use_disfluency_filter: bool
    use_profanity_filter: bool
    keywords: List[Tuple[str, float]]


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: RTZRModels = "sommers_ko",
        language: RTZRLanguages = "ko-KR",
        sample_rate: int = 16000,
        use_itn: bool = True,
        use_disfluency_filter: bool = False,
        use_profanity_filter: bool = False,
        keywords: List[Tuple[str, float]] = [],
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        """
        Create a new instance of RTZR STT.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        # Validate model/language
        if not isinstance(model, (str, RTZRModels)):
            raise ValueError(f"Invalid model type: {type(model)}")
        if not isinstance(language, (str, RTZRLanguages)):
            raise ValueError(f"Invalid language type: {type(language)}")

        # Validate sample rate
        if sample_rate not in [8000, 16000]:
            raise ValueError("Sample rate must be 8000 or 16000 Hz")

        client_id = client_id or os.environ.get("RTZR_CLIENT_ID")
        client_secret = client_secret or os.environ.get("RTZR_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "Must provide RTZR credentials (args or RTZR_CLIENT_ID/RTZR_CLIENT_SECRET env)."
            )

        self._client_id = client_id
        self._client_secret = client_secret
        self._session = requests.Session()
        self._token = None
        self._token_expire = 0
        self._streams = weakref.WeakSet[SpeechStream]()

        # Create shared gRPC channel with optimized settings
        self._channel = grpc.aio.secure_channel(
            GRPC_SERVER_URL,
            credentials=grpc.ssl_channel_credentials(),
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.enable_retries", 0),
            ],
        )

        # Create shared stub
        self._stub = pb_grpc.OnlineDecoderStub(self._channel)

        self._config = STTOptions(
            language=language,
            model=model,
            sample_rate=sample_rate,
            num_channels=1,
            use_itn=use_itn,
            use_disfluency_filter=use_disfluency_filter,
            use_profanity_filter=use_profanity_filter,
            keywords=keywords,
        )

    def _get_token(self) -> str:
        """Get or refresh the auth token."""
        try:
            # if no token or expired => fetch
            if not self._token or time.time() >= self._token_expire:
                resp = self._session.post(
                    f"{API_BASE}/v1/authenticate",
                    data={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                self._token_expire = data["expire_at"]
            return self._token
        except requests.exceptions.RequestException as e:
            raise APIConnectionError("Failed to fetch RTZR token") from e

    def stream(
        self,
        *,
        language: RTZRLanguages | str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            config=config,
            conn_options=conn_options,
            token_getter=self._get_token,
        )
        self._streams.add(stream)
        return stream

    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        config = dataclasses.replace(self._config)
        if language:
            config.language = language
        return config

    def update_options(
        self,
        *,
        language: RTZRLanguages | None = None,
        model: RTZRModels | None = None,
        use_itn: bool | None = None,
        use_disfluency_filter: bool | None = None,
        use_profanity_filter: bool | None = None,
        keywords: List[Tuple[str, float]] | None = None,
    ):
        if language is not None:
            self._config.language = language
        if model is not None:
            self._config.model = model
        if use_itn is not None:
            self._config.use_itn = use_itn
        if use_disfluency_filter is not None:
            self._config.use_disfluency_filter = use_disfluency_filter
        if use_profanity_filter is not None:
            self._config.use_profanity_filter = use_profanity_filter
        if keywords is not None:
            self._config.keywords = keywords

        # Also update existing streams
        for stream in self._streams:
            stream.update_options(
                language=language,
                model=model,
                use_itn=use_itn,
                use_disfluency_filter=use_disfluency_filter,
                use_profanity_filter=use_profanity_filter,
                keywords=keywords,
            )

    async def _recognize_impl(
        self,
        audio_data: bytes,
        language: str | None = None,
    ) -> List[stt.SpeechData]:
        # Non-streaming recognition is not implemented
        raise NotImplementedError("RTZR plugin supports streaming only.")

    async def close(self):
        """Close the STT instance and all resources."""
        for stream in list(self._streams):
            await stream.close()
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
        self._session.close()
        await super().close()


def _validate_keywords(keywords: List[Tuple[str, float]]) -> List[str]:
    """Convert (word, weight) -> 'word:weight' and do basic checks."""
    formatted = []
    for word, boost in keywords:
        # Example validation: only allow Korean & spaces. Adjust as needed
        if not all(c.isspace() or (0xAC00 <= ord(c) <= 0xD7A3) for c in word):
            raise ValueError(f"Keyword '{word}' must contain only Korean text/spaces")
        if len(word) > 20:
            raise ValueError(f"Keyword '{word}' is too long (max 20 chars)")

        if boost is None:
            formatted.append(word)
        else:
            if not (-5.0 <= boost <= 5.0):
                raise ValueError("Keyword boost must be between -5.0 and 5.0")
            formatted.append(f"{word}:{boost}")

    if len(formatted) > 100:
        raise ValueError("Too many keywords (max 100)")

    return formatted


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        config: STTOptions,
        conn_options: APIConnectOptions,
        token_getter: callable,
    ) -> None:
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=config.sample_rate
        )
        self._config = config
        self._token_getter = token_getter
        self._closed = False

    def update_options(
        self,
        *,
        language: RTZRLanguages | None = None,
        model: RTZRModels | None = None,
        use_itn: bool | None = None,
        use_disfluency_filter: bool | None = None,
        use_profanity_filter: bool | None = None,
        keywords: List[Tuple[str, float]] | None = None,
    ):
        if language is not None:
            self._config.language = language
        if model is not None:
            self._config.model = model
        if use_itn is not None:
            self._config.use_itn = use_itn
        if use_disfluency_filter is not None:
            self._config.use_disfluency_filter = use_disfluency_filter
        if use_profanity_filter is not None:
            self._config.use_profanity_filter = use_profanity_filter
        if keywords is not None:
            self._config.keywords = keywords

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            logger.warning("Trying to push_frame on a closed RTZR stream.")
            return
        super().push_frame(frame)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._input_ch:
                await self._input_ch.close()
            await asyncio.sleep(0)  # let tasks drain
        finally:
            await super().close()

    async def _run(self) -> None:
        try:

            async def request_iterator():
                try:
                    # Send initial config
                    decoder_config = pb.DecoderConfig(
                        sample_rate=self._config.sample_rate,
                        encoding=pb.DecoderConfig.AudioEncoding.LINEAR16,
                        model_name=self._config.model,
                        use_itn=self._config.use_itn,
                        use_disfluency_filter=self._config.use_disfluency_filter,
                        use_profanity_filter=self._config.use_profanity_filter,
                        keywords=_validate_keywords(self._config.keywords),
                    )
                    yield pb.DecoderRequest(streaming_config=decoder_config)

                    # Stream audio frames immediately without buffering
                    async for frame in self._input_ch:
                        if isinstance(frame, rtc.AudioFrame):
                            try:
                                # Send frame data directly without size limit
                                yield pb.DecoderRequest(
                                    audio_content=frame.data.tobytes()
                                )
                            finally:
                                frame = None
                except Exception as e:
                    logger.error(f"Error in request_iterator: {e}")
                    raise

            # Use shared stub with token credentials
            cred = grpc.access_token_call_credentials(self._token_getter())

            # Process responses
            async for response in self._stt._stub.Decode(
                request_iterator(), credentials=cred
            ):
                if response and response.results:
                    for result in response.results:
                        if not result.alternatives:
                            continue

                        alt = result.alternatives[0]
                        if not alt.text:
                            continue

                        # Create and emit speech event immediately
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=(
                                    stt.SpeechEventType.FINAL_TRANSCRIPT
                                    if result.is_final
                                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                                ),
                                alternatives=[
                                    stt.SpeechData(
                                        language=self._config.language,
                                        start_time=0,
                                        end_time=0,
                                        confidence=getattr(alt, "confidence", 1.0),
                                        text=alt.text,
                                    )
                                ],
                            )
                        )

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise APIStatusError("Authentication failed", status_code=401)
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise APIStatusError("Invalid parameters", status_code=400)
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise APIStatusError("Resource exhausted", status_code=429)
            elif e.code() == grpc.StatusCode.INTERNAL:
                raise APIStatusError("Internal server error", status_code=500)
            else:
                raise APIConnectionError() from e
        except Exception as e:
            raise APIConnectionError() from e
