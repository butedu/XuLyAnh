#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def main():
    # Kiểm tra config file tồn tại
    config_path = Path("config/train_config.yaml")
    if not config_path.exists():
        print(f" Không tìm thấy config file: {config_path}")
        print("   Vui lòng tạo file config hoặc chỉ định đường dẫn khác với --config")
        return 1
    
    print("="*60)
    print(" Smile Detection Training Script")
    print("="*60)
    
    # Kiểm tra GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            print(f" CUDA: {torch.version.cuda}")
        else:
            print("⚠️  Không phát hiện GPU, sẽ train trên CPU (chậm)")
    except ImportError:
        print("⚠️  PyTorch chưa được cài đặt")
    
    print("="*60)
    print()
    
    # Chạy training
    cmd = [sys.executable, "-m", "src.training.train"]
    
    # Thêm arguments từ command line
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    else:
        # Mặc định dùng config file
        cmd.extend(["--config", str(config_path)])
    
    print(f" Lệnh: {' '.join(cmd)}\n")
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
