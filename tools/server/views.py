import io
import os
import re
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Annotated

import soundfile as sf
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger

from fish_speech.utils.schema import (
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    UpdateReferenceResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_ ]+$")

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """Generate speech from text with the MLX backend."""
    try:
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.sample_rate

        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )

        audio = next(inference(req, engine))
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=req.format)

        return StreamResponse(
            iterable=buffer_to_async_generator(buffer.getvalue()),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in TTS generation: {exc}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """Add a reusable reference voice for later /v1/tts requests."""
    temp_file_path = None

    try:
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")
        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        model_manager: ModelManager = request.app.state.model_manager
        engine = model_manager.tts_inference_engine

        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        suffix = Path(getattr(audio, "filename", "") or "").suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as exc:
        logger.warning(f"Reference ID '{id}' already exists: {exc}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)

    except ValueError as exc:
        logger.warning(f"Invalid input for reference '{id}': {exc}")
        response = AddReferenceResponse(success=False, message=str(exc), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as exc:
        logger.error(f"File system error for reference '{id}': {exc}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as exc:
        logger.error(f"Unexpected error adding reference '{id}': {exc}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as exc:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {exc}"
                )


@routes.http.get("/v1/references/list")
async def list_references():
    """List reusable reference voice IDs."""
    try:
        model_manager: ModelManager = request.app.state.model_manager
        engine = model_manager.tts_inference_engine

        reference_ids = engine.list_reference_ids()
        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as exc:
        logger.error(f"Unexpected error listing references: {exc}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """Delete a reusable reference voice by ID."""
    try:
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")
        if not ID_PATTERN.match(reference_id) or len(reference_id) > 255:
            raise ValueError("Reference ID contains invalid characters or is too long")

        model_manager: ModelManager = request.app.state.model_manager
        engine = model_manager.tts_inference_engine
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as exc:
        logger.warning(f"Reference ID '{reference_id}' not found: {exc}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as exc:
        logger.warning(f"Invalid input for reference '{reference_id}': {exc}")
        response = DeleteReferenceResponse(
            success=False, message=str(exc), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as exc:
        logger.error(f"File system error deleting reference '{reference_id}': {exc}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as exc:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {exc}",
            exc_info=True,
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """Rename a reusable reference voice ID."""
    try:
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        if not ID_PATTERN.match(old_reference_id) or len(old_reference_id) > 255:
            raise ValueError(
                "Old reference ID contains invalid characters or is too long"
            )
        if not ID_PATTERN.match(new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        model_manager: ModelManager = request.app.state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        old_dir.rename(new_dir)
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' "
                f"to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as exc:
        logger.warning(str(exc))
        response = UpdateReferenceResponse(
            success=False,
            message=str(exc),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as exc:
        logger.warning(f"Invalid input for update reference: {exc}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(exc),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as exc:
        logger.error(f"File system error renaming reference: {exc}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as exc:
        logger.error(f"Unexpected error updating reference: {exc}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)
