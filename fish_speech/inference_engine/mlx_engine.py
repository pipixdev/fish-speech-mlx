"""
MLX backend for TTSInferenceEngine.

This module provides an alternative inference engine that uses mlx_audio
instead of the PyTorch-based stack (LLaMA + DAC).  The public interface
(the ``inference`` generator) is identical to the original
``TTSInferenceEngine`` so that the rest of the server / WebUI code needs
no further changes.

Usage
-----
Instantiate ``MLXTTSInferenceEngine`` wherever you would normally build a
``TTSInferenceEngine``, then pass it to the same server / WebUI code.

The engine wraps the reference-audio management helpers from
``ReferenceLoader`` so that the ``/v1/references/*`` API endpoints continue
to work when the MLX backend is selected.

mlx_audio API used
------------------
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model

    model = load_model(model_path)
    generate_audio(
        model=model,
        text=text,
        ref_audio=ref_audio_path,   # path to a WAV/MP3 file
        ref_text=ref_text,
        lang_code="auto",           # detected automatically when omitted
        file_prefix="output",
        output_path=output_dir,
        audio_format="wav",
        verbose=False,
    )
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.utils.schema import ServeTTSRequest

# ---------------------------------------------------------------------------
# Lazy import guard: mlx_audio is only available on Apple Silicon.
# We import inside the class methods so that importing this module on
# non-Apple platforms does not raise an ImportError at startup.
# ---------------------------------------------------------------------------

_MLX_AUDIO_AVAILABLE: bool | None = None


def _check_mlx_audio() -> bool:
    global _MLX_AUDIO_AVAILABLE
    if _MLX_AUDIO_AVAILABLE is None:
        try:
            import mlx_audio  # noqa: F401

            _MLX_AUDIO_AVAILABLE = True
        except ImportError:
            _MLX_AUDIO_AVAILABLE = False
    return _MLX_AUDIO_AVAILABLE


# Hard-coded default – overridden via ``MLXTTSInferenceEngine(model_path=…)``
DEFAULT_MLX_MODEL_PATH = "mlx-community/fish-audio-s2-pro-bf16"


class MLXTTSInferenceEngine(ReferenceLoader):
    """
    Drop-in replacement for ``TTSInferenceEngine`` that delegates all
    speech synthesis to ``mlx_audio``.

    Parameters
    ----------
    model_path:
        HuggingFace repo id or local path understood by
        ``mlx_audio.tts.utils.load_model``.
    sample_rate:
        Sample rate reported to callers.  mlx_audio produces 44 100 Hz by
        default for the fish-audio models.
    lang_code:
        BCP-47 language code forwarded to ``generate_audio`` (e.g. ``"ja"``,
        ``"zh"``, ``"en"``).  ``"auto"`` lets the model detect the language.
    """

    # MLX engine does not use a PyTorch decoder_model, but we expose
    # ``sample_rate`` as a plain attribute so that views.py can read
    # ``engine.decoder_model.sample_rate`` **OR** we patch decoder_model
    # below with a tiny shim.

    class _DecoderModelShim:
        """Minimal shim that satisfies ``engine.decoder_model.sample_rate``."""

        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        @property
        def device(self):
            # Return a CPU-like object so existing guard code does not crash.
            import types

            d = types.SimpleNamespace(type="cpu")
            return d

    def __init__(
        self,
        model_path: str = DEFAULT_MLX_MODEL_PATH,
        sample_rate: int = 44100,
        lang_code: str = "auto",
    ) -> None:
        super().__init__()

        if not _check_mlx_audio():
            raise ImportError(
                "mlx_audio is not installed. "
                "Install it with:  pip install mlx-audio"
            )

        from mlx_audio.tts.utils import load_model  # type: ignore[import]

        logger.info(f"[MLX] Loading model from {model_path!r} …")
        self._mlx_model = load_model(model_path)
        logger.info("[MLX] Model loaded.")

        self.sample_rate = sample_rate
        self.lang_code = lang_code

        # Provide a shim so that code doing ``engine.decoder_model.sample_rate``
        # (e.g. views.py line 138) keeps working without modification.
        self.decoder_model = self._DecoderModelShim(sample_rate)

        # Per-instance temp directory; created lazily on first use so that
        # merely importing the module does not touch the filesystem.
        self._tmp_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public interface – mirrors TTSInferenceEngine.inference()
    # ------------------------------------------------------------------

    def _ensure_tmp_dir(self) -> Path:
        """Return (and lazily create) the per-instance temp directory."""
        if self._tmp_dir is None or not self._tmp_dir.exists():
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="fish_mlx_"))
        return self._tmp_dir

    def inference(
        self, req: ServeTTSRequest
    ) -> Generator[InferenceResult, None, None]:
        """
        Generate speech using mlx_audio and yield ``InferenceResult`` objects
        with the same ``code`` values ("header", "segment", "final", "error")
        as the original engine so that all downstream consumers work unchanged.
        """
        from mlx_audio.tts.generate import generate_audio  # type: ignore[import]

        tmp_ref_path: str | None = None
        try:
            # ---- resolve reference audio --------------------------------
            ref_audio_path, ref_text, tmp_ref_path = self._resolve_reference(req)

            # ---- streaming header (WAV only) ----------------------------
            if req.streaming:
                from fish_speech.inference_engine.utils import wav_chunk_header

                yield InferenceResult(
                    code="header",
                    audio=(
                        self.sample_rate,
                        np.array(wav_chunk_header(sample_rate=self.sample_rate)),
                    ),
                    error=None,
                )

            # ---- synthesise --------------------------------------------
            with tempfile.TemporaryDirectory(dir=self._ensure_tmp_dir()) as tmp_out:
                generate_audio(
                    model=self._mlx_model,
                    text=req.text,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    lang_code=self.lang_code if self.lang_code != "auto" else None,
                    file_prefix="out",
                    output_path=tmp_out,
                    audio_format="wav",
                    verbose=False,
                )

                # mlx_audio writes ``<output_path>/out.wav`` (or with index)
                wav_files = sorted(Path(tmp_out).glob("*.wav"))
                if not wav_files:
                    raise RuntimeError(
                        "[MLX] generate_audio produced no output files."
                    )

                # Concatenate if multiple segments were written.
                segments = []
                for wf in wav_files:
                    data, sr = sf.read(str(wf), dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)  # mix down to mono
                    segments.append(data)

                audio = np.concatenate(segments, axis=0) if segments else np.array([])

        except Exception as exc:
            logger.error(f"[MLX] Inference error: {exc}", exc_info=True)
            yield InferenceResult(code="error", audio=None, error=exc)
            return
        finally:
            # Clean up any temp reference file written for this request.
            if tmp_ref_path is not None:
                try:
                    os.unlink(tmp_ref_path)
                except OSError:
                    pass

        yield InferenceResult(
            code="final",
            audio=(self.sample_rate, audio),
            error=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_reference(
        self, req: ServeTTSRequest
    ) -> tuple[str | None, str, str | None]:
        """
        Return ``(ref_audio_path, ref_text, tmp_path_to_cleanup)`` for the request.

        The third element is the path of any temporary file written for this
        request; the caller is responsible for deleting it after synthesis.

        Priority:
        1. ``req.reference_id``  → look up in ``references/`` directory
        2. ``req.references``    → first entry (bytes) → write to temp file
        3. Neither               → return (None, "", None)
        """
        if req.reference_id:
            ref_dir = Path("references") / req.reference_id
            audio_files = [
                f
                for f in ref_dir.iterdir()
                if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
            ]
            if not audio_files:
                raise FileNotFoundError(
                    f"[MLX] No audio found in references/{req.reference_id}"
                )
            audio_path = str(audio_files[0])
            lab_file = audio_files[0].with_suffix(".lab")
            ref_text = lab_file.read_text(encoding="utf-8") if lab_file.exists() else ""
            return audio_path, ref_text, None  # persistent file – don't delete

        if req.references:
            ref = req.references[0]
            # Sniff format from magic bytes so miniaudio can decode correctly.
            audio_bytes = ref.audio
            if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
                suffix = ".mp3"
            elif audio_bytes[:4] == b"fLaC":
                suffix = ".flac"
            elif audio_bytes[:4] == b"OggS":
                suffix = ".ogg"
            else:
                suffix = ".wav"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=self._ensure_tmp_dir()
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            return tmp_path, ref.text or "", tmp_path  # caller must delete tmp_path

        return None, "", None
