from .stt import STT
from .models import RTZRLanguages, RTZRModels
from .version import __version__
from . import vito_stt_client_pb2
from . import vito_stt_client_pb2_grpc

__all__ = [
    "STT",
    "RTZRLanguages",
    "RTZRModels",
    "__version__",
    "vito_stt_client_pb2",
    "vito_stt_client_pb2_grpc",
]

from livekit.agents import Plugin

from .log import logger


class RTZRPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(RTZRPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
