"""Tệp giữ lại để tránh lỗi import; dịch vụ nâng cao đã bị loại bỏ."""

class DichVuNhanDienCuoiNangCao:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Dịch vụ nâng cao đã bị loại bỏ. Vui lòng dùng SmileCounter mặc định.")
