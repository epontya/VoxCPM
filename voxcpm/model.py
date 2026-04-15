"""VoxCPM model loading and inference utilities."""

import os
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class VoxCPMModel:
    """Wrapper class for the VoxCPM speech recognition model.

    Handles model initialization, loading from local directories or
    HuggingFace Hub, and running inference on audio inputs.
    """

    DEFAULT_MODEL_ID = "openbmb/MiniCPM-o-2_6"
    SUPPORTED_SAMPLE_RATES = [16000, 22050, 44100]
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
        dtype: str = "auto",
    ):
        """
        Initialize the VoxCPM model wrapper.

        Args:
            model_dir: Path to local model directory or HuggingFace model ID.
                       Defaults to the official MiniCPM-o model.
            device: Device to run the model on ('cpu', 'cuda', 'auto').
            dtype: Data type for model weights ('float16', 'bfloat16', 'float32', 'auto').
        """
        self.model_dir = str(model_dir) if model_dir else self.DEFAULT_MODEL_ID
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        if self._is_loaded:
            logger.debug("Model already loaded, skipping.")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Required dependencies not found. "
                "Install with: pip install torch transformers"
            ) from e

        logger.info("Loading VoxCPM model from: %s", self.model_dir)

        # Resolve dtype
        torch_dtype = self._resolve_torch_dtype()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "auto" else "auto",
        )
        self._model.eval()
        self._is_loaded = True
        logger.info("Model loaded successfully.")

    def _resolve_torch_dtype(self):
        """Resolve the torch dtype from string configuration."""
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return dtype_map.get(self.dtype, torch.bfloat16)

    def transcribe(self, audio_path: Union[str, Path], prompt: Optional[str] = None) -> str:
        """Transcribe speech from an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).
            prompt: Optional text prompt to guide transcription.

        Returns:
            Transcribed text string.
        """
        if not self._is_loaded:
            self.load()

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.debug("Transcribing audio: %s", audio_path)

        msgs = self._build_messages(audio_path, prompt)
        result = self._model.chat(
            msgs=msgs,
            tokenizer=self._tokenizer,
        )
        return result.strip()

    def _build_messages(self, audio_path: str, prompt: Optional[str]) -> list:
        """Build the message list for model inference."""
        user_content = [
            {"type": "audio", "audio": audio_path},
        ]
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content.append({"type": "text", "text": "Please transcribe the audio."})

        return [{"role": "user", "content": user_content}]

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has been loaded."""
        return self._is_loaded

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"VoxCPMModel(model_dir='{self.model_dir}', device='{self.device}', status={status})"
