import re
import shutil
from pathlib import Path

from loguru import logger

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_ ]+$")


class ReferenceLoader:
    def __init__(self) -> None:
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

    @staticmethod
    def _validate_id(id: str) -> None:
        if not _ID_PATTERN.match(id) or len(id) > 255:
            raise ValueError(
                "Reference ID contains invalid characters or is too long. "
                "Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )

    @staticmethod
    def _audio_files(ref_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in ref_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

    def list_reference_ids(self) -> list[str]:
        ref_base_path = Path("references")
        if not ref_base_path.exists():
            return []

        valid_ids = []
        for ref_dir in ref_base_path.iterdir():
            if not ref_dir.is_dir():
                continue

            for audio_file in self._audio_files(ref_dir):
                if audio_file.with_suffix(".lab").exists():
                    valid_ids.append(ref_dir.name)
                    break

        return sorted(valid_ids)

    def add_reference(self, id: str, audio_file_path: str, reference_text: str) -> None:
        self._validate_id(id)

        ref_dir = Path("references") / id
        if ref_dir.exists():
            raise FileExistsError(f"Reference ID '{id}' already exists")

        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
            )

        try:
            ref_dir.mkdir(parents=True, exist_ok=False)
            target_audio_path = ref_dir / f"sample{audio_path.suffix.lower()}"
            shutil.copy2(audio_path, target_audio_path)
            (ref_dir / "sample.lab").write_text(reference_text, encoding="utf-8")
            self.ref_by_id.pop(id, None)
            logger.info(f"Successfully added reference voice with ID: {id}")
        except Exception:
            if ref_dir.exists():
                shutil.rmtree(ref_dir)
            raise

    def delete_reference(self, id: str) -> None:
        self._validate_id(id)

        ref_dir = Path("references") / id
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        try:
            shutil.rmtree(ref_dir)
            self.ref_by_id.pop(id, None)
            logger.info(f"Successfully deleted reference voice with ID: {id}")
        except Exception as exc:
            logger.error(f"Failed to delete reference '{id}': {exc}")
            raise OSError(f"Failed to delete reference '{id}': {exc}") from exc
