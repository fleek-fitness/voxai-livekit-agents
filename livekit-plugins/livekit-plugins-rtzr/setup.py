import os
import pathlib

import setuptools
import setuptools.command.build_py

here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "plugins", "rtzr", "version.py"), "r") as f:
    exec(f.read(), about)

setuptools.setup(
    name="livekit-plugins-rtzr",
    version=about["__version__"],
    description="LiveKit Agents Plugin for RTZR STT",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["webrtc", "realtime", "audio", "livekit", "stt", "rtzr"],
    license="Apache-2.0",
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9.0",
    install_requires=[
        "livekit-agents>=0.12.3",
        "grpcio==1.67.1",
        "grpcio-tools==1.67.1",
        "requests>=2.31.0",
        "protobuf>=4.24.0",
        "pydub>=0.25.1",
    ],
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
