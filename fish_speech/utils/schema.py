import base64
from typing import Annotated, Literal

from pydantic import BaseModel, Field, conint, model_validator


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if isinstance(audio, str) and len(audio) > 255:
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=1000, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3", "opus"] = "wav"
    latency: Literal["normal", "balanced"] = "normal"
    references: list[ServeReferenceAudio] = Field(default_factory=list)
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8


class AddReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class ListReferencesResponse(BaseModel):
    success: bool
    reference_ids: list[str]
    message: str = "Success"


class DeleteReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class UpdateReferenceResponse(BaseModel):
    success: bool
    message: str
    old_reference_id: str
    new_reference_id: str
