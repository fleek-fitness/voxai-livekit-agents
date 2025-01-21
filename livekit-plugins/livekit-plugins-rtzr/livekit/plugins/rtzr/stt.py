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

        Args:
            model: The RTZR model to use
            language: The language code (e.g., "ko-KR")
            sample_rate: Audio sample rate in Hz
            use_itn: Enable inverse text normalization
            use_disfluency_filter: Enable disfluency filter
            use_profanity_filter: Enable profanity filter
            keywords: List of keyword tuples (word, boost)
            client_id: RTZR client ID
            client_secret: RTZR client secret
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        # Validate model and language
        if not isinstance(model, (str, RTZRModels)):
            raise ValueError(f"Invalid model type: {type(model)}")
        if not isinstance(language, (str, RTZRLanguages)):
            raise ValueError(f"Invalid language type: {type(language)}")

        # Validate sample rate
        if sample_rate not in [8000, 16000]:
            raise ValueError("Sample rate must be either 8000 or 16000 Hz")

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
        self._streams = weakref.WeakSet[SpeechStream]()

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
        """Get or refresh the authentication token."""
        try:
            if not self._token or time.time() >= self._token_expire:
                resp = self._session.post(
                    f"{API_BASE}/v1/authenticate",
                    data={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    timeout=10,  # Add timeout
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                self._token_expire = data["expire_at"]
            return self._token
        except requests.exceptions.RequestException as e:
            raise APIConnectionError("Failed to get authentication token") from e

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

    async def close(self):
        """Close the STT instance and all active streams."""
        for stream in list(self._streams):
            await stream.close()
        self._session.close()
        await super().close()


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
        self._closed = False
        self._bytes_per_sample = 2

    async def _process_result(self, result) -> None:
        """Process a single result from the RTZR API."""
        if not result.alternatives:
            return

        # Convert RTZR result to SpeechData format
        speech_data = stt.SpeechData(
            language=self._config.language,
            start_time=0,  # RTZR doesn't provide word-level timing
            end_time=0,  # RTZR doesn't provide word-level timing
            confidence=(
                result.alternatives[0].confidence
                if hasattr(result.alternatives[0], "confidence")
                else 1.0
            ),
            text=result.alternatives[0].text,
        )

        # Use _emit instead of emit_speech
        await self._emit(speech_data)

    async def _run(self) -> None:
        channel = None
        try:
            channel = grpc.aio.secure_channel(
                GRPC_SERVER_URL,
                credentials=grpc.ssl_channel_credentials(),
            )

            from . import vito_stt_client_pb2 as pb
            from . import vito_stt_client_pb2_grpc as pb_grpc

            stub = pb_grpc.OnlineDecoderStub(channel)

            async def request_iterator():
                try:
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

                    async for frame in self._input_ch:
                        if isinstance(frame, rtc.AudioFrame):
                            try:
                                data = frame.data.tobytes()
                                if len(data) > 1024 * 1024:
                                    data = data[: 1024 * 1024]
                                yield pb.DecoderRequest(audio_content=data)
                            finally:
                                frame = None
                except Exception as e:
                    logger.error(f"Error in request_iterator: {e}")
                    raise

            cred = grpc.access_token_call_credentials(self._token_getter())

            try:
                async for response in stub.Decode(request_iterator(), credentials=cred):
                    if response and response.results:
                        for result in response.results:
                            try:
                                await self._process_result(result)
                            except Exception as e:
                                logger.error(f"Error processing result: {e}")
                            finally:
                                del result
                    del response
            except asyncio.CancelledError:
                logger.debug("Stream cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in response processing: {e}")
                raise

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise APIStatusError("Authentication failed", status_code=401)
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise APIStatusError("Invalid parameters", status_code=400)
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise APIStatusError(
                    "Resource exhausted or payment required", status_code=429
                )
            elif e.code() == grpc.StatusCode.INTERNAL:
                raise APIStatusError("Internal server error", status_code=500)
            else:
                raise APIConnectionError() from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    logger.error(f"Error closing channel: {e}")

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

        self._closed = False

    async def close(self):
        """Properly close the stream."""
        if not self._closed:
            self._closed = True
            try:
                if self._input_ch:
                    await self._input_ch.close()
                # Wait for any pending operations to complete
                await asyncio.sleep(0)
            finally:
                await super().close()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Override push_frame to check closed state."""
        if self._closed:
            logger.warning("Attempting to push frame to closed stream")
            return
        super().push_frame(frame)

    def _check_not_closed(self) -> None:
        """Override to use our closed flag."""
        if self._closed:
            cls = self.__class__
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")
