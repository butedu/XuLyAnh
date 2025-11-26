"""Backend FastAPI cho ứng dụng demo phát hiện nụ cười."""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.service import DichVuNhanDienCuoi

app = FastAPI(title="Demo Phát Hiện Nụ Cười", version="0.1.0")

# Thư mục chứa giao diện frontend
THU_MUC_STATIC = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=THU_MUC_STATIC), name="static")

dich_vu: DichVuNhanDienCuoi | None = None


@app.on_event("startup")
async def khoi_dong() -> None:
    """Khởi động ứng dụng và tải mô hình."""
    global dich_vu
    try:
        dich_vu = DichVuNhanDienCuoi()
    except FileNotFoundError as exc:
        # Trì hoãn lỗi đến request đầu tiên để UI có thể hiển thị hướng dẫn
        dich_vu = None
        print(f"[khoi_dong] Cảnh báo: {exc}")


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
        raise HTTPException(
            status_code=500,
            detail="Mô hình chưa được tải. Vui lòng huấn luyện mô hình và đặt file trọng số trong models/",
        )
    
    try:
        # Phân tích ảnh
        ket_qua_phan_tich = dich_vu.phan_tich_anh_bytes(du_lieu)
        # Tạo ảnh chú thích
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
