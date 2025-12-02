"""Bộ gom detector cũ đã bị loại bỏ."""


class EnsembleFaceDetector:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("EnsembleFaceDetector đã bị loại bỏ. Dùng YOLOFaceDetector trong src.detection.")
