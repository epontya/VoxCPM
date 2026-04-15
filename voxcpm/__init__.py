"""VoxCPM: A speech recognition and processing toolkit based on OpenBMB/VoxCPM.

This package provides utilities for loading and running VoxCPM models
for automatic speech recognition (ASR) tasks.

Note: This is a personal fork. Main changes from upstream:
- Added __author_email__ metadata
- Exposed __version__ more prominently in __all__
"""

__version__ = "0.1.0"
__author__ = "VoxCPM Contributors"
__author_email__ = ""  # placeholder for personal fork contact
__license__ = "Apache-2.0"

from voxcpm.model import VoxCPMModel
from voxcpm.processor import AudioProcessor

__all__ = [
    "VoxCPMModel",
    "AudioProcessor",
    "__version__",
    "__author__",
]
