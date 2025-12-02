"""MTCNN detector đã bị loại bỏ khỏi dự án."""


class MTCNNFaceDetector:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("MTCNNFaceDetector không còn được hỗ trợ. Hãy dùng YOLOFaceDetector từ src.detection.")
