"""Backend FastAPI cho pipeline YOLO + CNN."""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.service import DichVuNhanDienCuoi

app = FastAPI(title="Demo Phát Hiện Nụ Cười", version="1.0.0")

THU_MUC_STATIC = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=THU_MUC_STATIC), name="static")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "video_results"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_RESULTS: dict[str, dict[str, object]] = {}
VIDEO_LOCK = asyncio.Lock()

dich_vu: DichVuNhanDienCuoi | None = None


@app.on_event("startup")
async def khoi_dong() -> None:
    """Tải mô hình khi khởi động."""

    global dich_vu
    model_path = PROJECT_ROOT / "models" / "smile_cnn_best.pth"
    face_model = PROJECT_ROOT / "models" / "yolov8n-face.pt"
    try:
        dich_vu = DichVuNhanDienCuoi(duong_dan_mo_hinh=str(model_path), duong_dan_face=str(face_model))
    except Exception as exc:  # noqa: BLE001
        dich_vu = None
        print(f"[khoi_dong] Không thể tải dịch vụ: {exc}")


@app.get("/", response_class=HTMLResponse)
async def trang_chu() -> str:
    """Hiển thị trang chủ."""
    duong_dan_index = THU_MUC_STATIC / "index.html"
    if not duong_dan_index.exists():
        raise HTTPException(status_code=404, detail="Thiếu file giao diện frontend")
    return duong_dan_index.read_text(encoding="utf-8")


@app.post("/api/detect")
async def phat_hien_nu_cuoi(file: UploadFile = File(...)) -> JSONResponse:
    """API endpoint để phát hiện nụ cười trong ảnh tải lên.
    
    Tham số:
        file: File ảnh từ client
        
    Trả về:
        JSON chứa kết quả phân tích và ảnh đã chú thích
    """
    # Kiểm tra loại file
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file ảnh")
    
    # Đọc dữ liệu file
    du_lieu = await file.read()
    if not du_lieu:
        raise HTTPException(status_code=400, detail="File rỗng")
    
    # Kiểm tra mô hình đã tải chưa
    if dich_vu is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng. Kiểm tra lại trọng số trong models/.")
    
    try:
        ket_qua_phan_tich = dich_vu.phan_tich_anh_bytes(du_lieu)
        anh_chu_thich_bytes = dich_vu.chu_thich_anh(du_lieu, ket_qua_phan_tich)
    except Exception as exc:  # noqa: BLE001 - hiển thị lỗi cho client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    # Mã hóa ảnh thành base64
    anh_ma_hoa = base64.b64encode(anh_chu_thich_bytes).decode("ascii")
    phan_hoi = {
        "summary": ket_qua_phan_tich,
        "annotated_image": f"data:image/jpeg;base64,{anh_ma_hoa}",
    }
    return JSONResponse(content=phan_hoi)


async def _cleanup_video_result(token: str, delay_seconds: int = 600) -> None:
    await asyncio.sleep(delay_seconds)
    async with VIDEO_LOCK:
        info = VIDEO_RESULTS.pop(token, None)
    if info:
        try:
            Path(info["path"]).unlink(missing_ok=True)
        except OSError:
            pass


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@app.post("/api/detect-video")
async def phat_hien_nu_cuoi_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    frame_skip: int = Form(0),
    resize_width: int | None = Form(None),
    resize_height: int | None = Form(None),
) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file video hợp lệ")

    if frame_skip < 0:
        raise HTTPException(status_code=400, detail="frame_skip phải lớn hơn hoặc bằng 0")

    if (resize_width is None) != (resize_height is None):
        raise HTTPException(status_code=400, detail="Cần cung cấp cả chiều rộng và chiều cao khi resize")

    if dich_vu is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng. Kiểm tra lại trọng số trong models/.")

    token = uuid4().hex
    input_suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    input_path = VIDEO_OUTPUT_DIR / f"{token}_input{input_suffix}"
    output_path = VIDEO_OUTPUT_DIR / f"{token}_annotated.mp4"

    try:
        with input_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="File video rỗng")

        resize = None
        if resize_width is not None and resize_height is not None:
            if resize_width <= 0 or resize_height <= 0:
                raise HTTPException(status_code=400, detail="Kích thước resize phải dương")
            resize = (int(resize_width), int(resize_height))

        stats = dich_vu.xu_ly_video_file(
            duong_dan_vao=input_path,
            duong_dan_ra=output_path,
            frame_skip=frame_skip,
            resize=resize,
        )
    except HTTPException:
        _safe_unlink(input_path)
        raise
    except Exception as exc:  # noqa: BLE001
        _safe_unlink(input_path)
        _safe_unlink(output_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background.add_task(_safe_unlink, input_path)

    async with VIDEO_LOCK:
        VIDEO_RESULTS[token] = {"path": output_path, "summary": stats}

    asyncio.create_task(_cleanup_video_result(token))

    return JSONResponse(
        content={
            "summary": stats,
            "token": token,
            "download_url": f"/api/video-results/{token}",
        }
    )


@app.get("/api/video-results/{token}")
async def tai_video_da_chu_thich(token: str) -> FileResponse:
    async with VIDEO_LOCK:
        info = VIDEO_RESULTS.get(token)

    if info is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy video đã xử lý hoặc đã hết hạn")

    duong_dan = Path(info["path"])
    if not duong_dan.exists():
        raise HTTPException(status_code=404, detail="Video đã xử lý không còn tồn tại")

    return FileResponse(
        path=duong_dan,
        media_type="video/mp4",
        filename=f"smile-counter-{token}.mp4",
    )


if __name__ == "__main__":
    # Chạy ứng dụng với Uvicorn khi gọi trực tiếp module này
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=False)
