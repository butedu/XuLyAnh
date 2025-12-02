"""Backend FastAPI cho pipeline YOLO + CNN."""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.service import DichVuNhanDienCuoi

app = FastAPI(title="Demo Phát Hiện Nụ Cười", version="1.0.0")

THU_MUC_STATIC = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=THU_MUC_STATIC), name="static")

dich_vu: DichVuNhanDienCuoi | None = None


@app.on_event("startup")
async def khoi_dong() -> None:
    """Tải mô hình khi khởi động."""

    global dich_vu
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / "models" / "smile_cnn_best.pth"
    face_model = project_root / "models" / "yolov8n-face.pt"
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


if __name__ == "__main__":
    # Chạy ứng dụng với Uvicorn khi gọi trực tiếp module này
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=False)
