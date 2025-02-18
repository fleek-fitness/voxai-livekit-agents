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
import json
import uuid
from typing import AsyncIterable

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APIError,
    stt,
    utils,
    vad,
)
from livekit.agents.utils import AudioBuffer
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.vad import VADEventType, VADEvent

# Message constants (example values)
# _KEEPALIVE_MSG = '{"type": "keepalive"}'
# _FINALIZE_MSG = '{"type": "finalize"}'
# _CLOSE_MSG = '{"type": "close"}'
_KEEPALIVE_MSG = "KEEPALIVE"
_FINALIZE_MSG = "FINALIZE"
_CLOSE_MSG = "CLOSE"


class STT(stt.STT):
    """
    Non-streaming STT implementation using aiohttp websockets.
    (The _recognize_impl method is omitted for brevity.)
    """

    def __init__(
        self,
        uri: str,
        vad: vad.VAD,
        *,
        capabilities: stt.STTCapabilities | None = None,
    ):
        if capabilities is None:
            capabilities = stt.STTCapabilities(streaming=True, interim_results=True)
        super().__init__(capabilities=capabilities)
        self.uri = uri
        self._vad = vad

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        sample_rate: int | None = None,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            vad=self._vad,
            conn_options=conn_options,
            sample_rate=sample_rate,
            language=language,
        )

    # _recognize_impl for non-streaming is omitted for brevity.
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        pass


class SpeechStream(stt.SpeechStream):
    """
    Streaming STT implementation using aiohttp websockets.
    Audio frames are sent as they are pushed into the stream.
    Transcription messages from the websocket are processed to emit SpeechEvents.
    Additionally, the VAD stream is monitored, and when an END_OF_SPEECH
    event is detected, a finalize message is sent via the websocket.
    """

    def __init__(
        self,
        *,
        stt: STT,
        vad: vad.VAD,
        conn_options: APIConnectOptions,
        sample_rate: int | None = None,
        language: str | None = None,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self.language = language
        self._vad = vad
        self._vad_stream = None  # Initialize as None, create in _run

    async def _dummy_vad_stream(self) -> AsyncIterable[VADEvent]:
        """
        Dummy VAD stream for demonstration.
        Replace this with your actual VAD event source.
        """
        await asyncio.sleep(1)
        yield VADEvent(type=VADEventType.START_OF_SPEECH)
        await asyncio.sleep(3)
        yield VADEvent(type=VADEventType.END_OF_SPEECH)

    async def _run(self) -> None:
        closing_ws = False
        self._vad_stream = self._vad.stream()

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse):
            try:
                while True:
                    await ws.send_str(_KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception:
                return

        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    # You might want to send a finalize message here if no VAD event occurs.
                    await ws.send_str(_FINALIZE_MSG)
                elif data is None:
                    break
                else:
                    await ws.send_bytes(data.data)
            closing_ws = True
            await ws.send_str(_CLOSE_MSG)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIError("Websocket closed unexpectedly", body=None)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    # Expect messages like:
                    # "{\"is_final\":false,\"language\":\"ko\",\"duration\":5.0,\"text\":\" ...\"}"
                    text = msg.data.strip()
                    data = json.loads(text)
                    if data.get("is_final", False):
                        event = SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=str(uuid.uuid4()),
                            alternatives=[
                                SpeechData(
                                    language=data.get(
                                        "language", self.language or "unknown"
                                    ),
                                    text=data.get("text", ""),
                                )
                            ],
                        )
                        self._event_ch.put_nowait(event)
                        break  # Final transcript received; exit recv loop.
                    else:
                        event = SpeechEvent(
                            type=SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=str(uuid.uuid4()),
                            alternatives=[
                                SpeechData(
                                    language=data.get(
                                        "language", self.language or "unknown"
                                    ),
                                    text=data.get("text", ""),
                                )
                            ],
                        )
                        self._event_ch.put_nowait(event)
                except Exception as e:
                    print("Failed to process message:", e)
                    continue

        async def vad_task(ws: aiohttp.ClientWebSocketResponse):
            # Process VAD events and forward them as SpeechEvents.
            async for vad_event in self._vad_stream:
                if vad_event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.put_nowait(
                        SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                elif vad_event.type == VADEventType.END_OF_SPEECH:
                    self._event_ch.put_nowait(
                        SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
                    )
                    # When end-of-speech is detected by VAD, send the finalize message.
                    await ws.send_str(_FINALIZE_MSG)

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self._stt.uri, timeout=self._conn_options.timeout
                    ) as ws:
                        tasks = [
                            asyncio.create_task(send_task(ws)),
                            asyncio.create_task(recv_task(ws)),
                            asyncio.create_task(keepalive_task(ws)),
                            asyncio.create_task(vad_task(ws)),
                        ]
                        done, _ = await asyncio.wait(
                            [asyncio.gather(*tasks)],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            task.result()
                        await utils.aio.gracefully_cancel(*tasks)
                        break
            finally:
                if ws is not None:
                    await ws.close()
