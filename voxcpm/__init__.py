"""VoxCPM: A speech recognition and processing toolkit based on OpenBMB/VoxCPM.

This package provides utilities for loading and running VoxCPM models
for automatic speech recognition (ASR) tasks.
"""

__version__ = "0.1.0"
__author__ = "VoxCPM Contributors"
__license__ = "Apache-2.0"

from voxcpm.model import VoxCPMModel
from voxcpm.processor import AudioProcessor

__all__ = [
    "VoxCPMModel",
    "AudioProcessor",
    "__version__",
]
