# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import weakref
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

        Credentials must be provided either through arguments or environment variables:
        - RTZR_CLIENT_ID
        - RTZR_CLIENT_SECRET
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        client_id = client_id or os.environ.get("RTZR_CLIENT_ID")
        client_secret = client_secret or os.environ.get("RTZR_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "RTZR credentials must be provided either through arguments or "
                "RTZR_CLIENT_ID and RTZR_CLIENT_SECRET environment variables"
            )

        self._client_id = client_id
        self._client_secret = client_secret
        self._token = None
        self._token_expire = 0
        self._session = requests.Session()

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
        self._streams = weakref.WeakSet[SpeechStream]()

    def _get_token(self) -> str:
        if not self._token or time.time() >= self._token_expire:
            resp = self._session.post(
                f"{API_BASE}/v1/authenticate",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expire = data["expire_at"]
        return self._token

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
        self, audio_data: bytes, language: str | None = None
    ) -> List[stt.SpeechData]:
        """
        Implements the abstract method for non-streaming recognition.
        This is required by the base STT class but RTZR primarily supports streaming.
        """
        raise NotImplementedError(
            "RTZR STT only supports streaming recognition. "
            "Use stream() method instead."
        )


def _validate_keywords(keywords: List[Tuple[str, float]]) -> List[str]:
    """Validate and format keywords according to RTZR specifications."""
    formatted_keywords = []
    for word, score in keywords:
        # Validate word format (Korean syllables and spaces only)
        if not all(c.isspace() or (0xAC00 <= ord(c) <= 0xD7A3) for c in word):
            raise ValueError(
                f"Keyword '{word}' must contain only Korean syllables and spaces"
            )

        # Check word length
        if len(word) > 20:
            raise ValueError(
                f"Keyword '{word}' exceeds maximum length of 20 characters"
            )

        # Validate score range
        if score is not None and not (-5.0 <= score <= 5.0):
            raise ValueError(f"Score for keyword '{word}' must be between -5.0 and 5.0")

        # Format keyword string
        if score is None:
            formatted_keywords.append(word)  # Will use default score of 2.0
        else:
            formatted_keywords.append(f"{word}:{score}")

    # Check total keywords limit
    if len(formatted_keywords) > 100:
        raise ValueError("Maximum number of keywords (100) exceeded")

    return formatted_keywords


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
            stt=stt,
            conn_options=conn_options,
            sample_rate=config.sample_rate,
        )
        self._config = config
        self._token_getter = token_getter
        self._reconnect_event = asyncio.Event()
        self._speaking = False

    async def _run(self) -> None:
        while True:
            try:
                channel = grpc.aio.secure_channel(
                    GRPC_SERVER_URL,
                    credentials=grpc.ssl_channel_credentials(),
                )

                # Import the generated GRPC code
                from . import vito_stt_client_pb2 as pb
                from . import vito_stt_client_pb2_grpc as pb_grpc

                stub = pb_grpc.OnlineDecoderStub(channel)

                async def request_iterator():
                    # First request with config
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

                    # Stream audio data
                    async for frame in self._input_ch:
                        if isinstance(frame, rtc.AudioFrame):
                            yield pb.DecoderRequest(audio_content=frame.data.tobytes())

                creds = grpc.access_token_call_credentials(self._token_getter())
                responses = stub.Decode(
                    request_iterator(),
                    credentials=grpc.composite_channel_credentials(
                        grpc.ssl_channel_credentials(), creds
                    ),
                )

                async for response in responses:
                    for result in response.results:
                        if not result.alternatives:
                            continue

                        # Detect start of speech if we have alternatives
                        if not self._speaking and result.alternatives[0].text.strip():
                            self._speaking = True
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.START_OF_SPEECH
                                )
                            )

                        event_type = (
                            stt.SpeechEventType.FINAL_TRANSCRIPT
                            if result.is_final
                            else stt.SpeechEventType.INTERIM_TRANSCRIPT
                        )

                        alternatives = [
                            stt.SpeechData(
                                language=self._config.language,
                                start_time=result.start_at / 1000,
                                end_time=(result.start_at + result.duration) / 1000,
                                confidence=alt.confidence,
                                text=alt.text,
                            )
                            for alt in result.alternatives
                        ]

                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=event_type,
                                alternatives=alternatives,
                            )
                        )

                        # If this was a final result, mark end of speech
                        if result.is_final and self._speaking:
                            self._speaking = False
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                            )

                if await self._reconnect_event.wait():
                    self._reconnect_event.clear()
                    continue
                break

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAUTHENTICATED:  # Code 16
                    raise APIStatusError("Authentication failed", status_code=401)
                elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:  # Code 3
                    raise APIStatusError("Invalid parameters", status_code=400)
                elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:  # Code 8
                    raise APIStatusError(
                        "Resource exhausted or payment required", status_code=429
                    )
                elif e.code() == grpc.StatusCode.INTERNAL:  # Code 13
                    raise APIStatusError("Internal server error", status_code=500)
                else:
                    raise APIConnectionError() from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                await channel.close()

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

        self._reconnect_event.set()
