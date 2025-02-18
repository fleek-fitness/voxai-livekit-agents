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
from .. import utils
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, APIError
from ..vad import VAD, VADEventType
from .stt import (
    STT,
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
    SpeechData,
)

# Message constants
_KEEPALIVE_MSG = "KEEPALIVE"
_START_MSG = "START"
_FINALIZE_MSG = "FINALIZE"
_CLOSE_MSG = "CLOSE"


class NewStreamAdapter(STT):
    def __init__(self, *, stt: STT, vad: VAD, uri: str) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=True)
        )
        self._vad = vad
        self._stt = stt
        self.uri = uri

        @self._stt.on("metrics_collected")
        def _forward_metrics(*args, **kwargs):
            self.emit("metrics_collected", *args, **kwargs)

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return NewStreamAdapterWrapper(
            self,
            vad=self._vad,
            wrapped_stt=self._stt,
            language=language,
            conn_options=conn_options,
            uri=self.uri,
        )


class NewStreamAdapterWrapper(RecognizeStream):
    def __init__(
        self,
        stt: STT,
        *,
        vad: VAD,
        wrapped_stt: STT,
        language: str | None,
        conn_options: APIConnectOptions,
        uri: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._vad = vad
        self._wrapped_stt = wrapped_stt
        self._vad_stream = self._vad.stream()
        self._language = language
        self._uri = uri

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SpeechEvent]
    ) -> None:
        pass  # do nothing

    async def _run(self) -> None:
        closing_ws = False

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
                    await ws.send_str(_FINALIZE_MSG)
                    self._vad_stream.flush()
                elif data is None:
                    break
                else:
                    if data.data is not None:
                        await ws.send_bytes(data.data.tobytes())
                    # self._vad_stream.push_frame(data)
            closing_ws = True
            await ws.send_str(_CLOSE_MSG)
            self._vad_stream.end_input()

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
                    text = msg.data.strip()
                    data = json.loads(text)
                    if data.get("is_final", False):
                        event = SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=str(uuid.uuid4()),
                            alternatives=[
                                SpeechData(
                                    language=data.get(
                                        "language", self._language or "unknown"
                                    ),
                                    text=data.get("text", ""),
                                )
                            ],
                        )
                        self._event_ch.send_nowait(event)
                    else:
                        event = SpeechEvent(
                            type=SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=str(uuid.uuid4()),
                            alternatives=[
                                SpeechData(
                                    language=data.get(
                                        "language", self._language or "unknown"
                                    ),
                                    text=data.get("text", ""),
                                )
                            ],
                        )
                        self._event_ch.send_nowait(event)
                except Exception as e:
                    print("Failed to process message:", e)
                    continue

        async def vad_task(ws: aiohttp.ClientWebSocketResponse):
            async for vad_event in self._vad_stream:
                if vad_event.type == VADEventType.START_OF_SPEECH:
                    await ws.send_str(_START_MSG)
                    self._event_ch.send_nowait(
                        SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                elif vad_event.type == VADEventType.END_OF_SPEECH:
                    await ws.send_str(_FINALIZE_MSG)
                    self._event_ch.send_nowait(
                        SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
                    )

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self._uri, timeout=self._conn_options.timeout
                    ) as ws:
                        tasks = [
                            asyncio.create_task(send_task(ws), name="send_task"),
                            asyncio.create_task(recv_task(ws), name="recv_task"),
                            asyncio.create_task(
                                keepalive_task(ws), name="keepalive_task"
                            ),
                            asyncio.create_task(vad_task(ws), name="vad_task"),
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
