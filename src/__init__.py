"""
Package gốc src/ — Dự án BTL Data Mining: Dự đoán Bệnh Tim (UCI Heart Disease)
"""

import os
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_params(path: str = None) -> dict:
    """Đọc file params.yaml và trả về dict cấu hình."""
    if path is None:
        path = os.path.join(ROOT_DIR, "configs", "params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def get_path(relative: str) -> str:
    """Chuyển đường dẫn tương đối thành tuyệt đối (từ ROOT_DIR)."""
    return os.path.join(ROOT_DIR, relative)


# Tự động tạo thư mục output nếu chưa có
for _d in ["outputs/figures", "outputs/tables", "outputs/models", "outputs/reports", "data/processed"]:
    os.makedirs(get_path(_d), exist_ok=True)
