"""Module nâng cao cũ đã bị vô hiệu hóa."""

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Smile Detection - Advanced", version="deprecated")


@app.get("/")
async def notify() -> None:  # pragma: no cover
    raise HTTPException(status_code=410, detail="Phiên bản nâng cao đã bị loại bỏ. Dùng webapp.backend.main.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Endpoint nâng cao đã bị loại bỏ. Dùng webapp.backend.main:app")
