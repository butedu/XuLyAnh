import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .advanced_service import DichVuNhanDienCuoiNangCao

app = FastAPI(title='Smile Detection - Advanced (MTCNN)')

# initialize service (will try to load models from models/)
service = DichVuNhanDienCuoiNangCao()


@app.post('/api/detect_advanced')
async def detect_advanced(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail='Uploaded file is not an image')
    contents = await file.read()
    tmp_path = os.path.join(ROOT, 'data', 'tmp_upload.png')
    with open(tmp_path, 'wb') as f:
        f.write(contents)
    try:
        out = service.phan_tich_tu_file(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return JSONResponse(content=out)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('webapp.backend.advanced_main:app', host='127.0.0.1', port=9000, reload=True)
