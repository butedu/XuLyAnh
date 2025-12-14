"""
Script test nhanh để kiểm tra config và model architecture.
Chạy trước khi training để đảm bảo mọi thứ hoạt động.

Usage: python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test tất cả imports cần thiết."""
    print("🔍 Testing imports...")
    try:
        # Import các thư viện cần cho training và xử lý dữ liệu
        import torch
        import torchvision
        import yaml
        import pandas
        import numpy
        from PIL import Image

        # Nếu import thành công
        print("   ✅ All imports successful")

        # In thông tin version PyTorch
        print(f"   PyTorch: {torch.__version__}")

        # Kiểm tra CUDA (GPU) có khả dụng không
        print(f"   CUDA available: {torch.cuda.is_available()}")

        # Nếu có GPU thì in tên GPU
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        return True
    except ImportError as e:
        # Lỗi nếu thiếu thư viện
        print(f"   ❌ Import error: {e}")
        return False


def test_config():
    """Test config file tồn tại và valid."""
    print("\n🔍 Testing config file...")

    # Đường dẫn file config training
    config_path = Path("config/train_config.yaml")
    
    # Kiểm tra file config có tồn tại không
    if not config_path.exists():
        print(f"   ❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml

        # Load file YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Các key bắt buộc phải có trong config
        required_keys = ['data', 'training', 'model', 'augmentation', 'settings']
        for key in required_keys:
            if key not in config:
                print(f"   ❌ Missing key in config: {key}")
                return False
        
        # In thông tin config quan trọng
        print("   ✅ Config file valid")
        print(f"   Model: {config['model'].get('name', 'SmileNet')}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   LR: {config['training']['learning_rate']}")
        return True
    except Exception as e:
        # Lỗi đọc hoặc parse config
        print(f"   ❌ Config error: {e}")
        return False


def test_model_architecture():
    """Test model có thể build được không."""
    print("\n🔍 Testing model architecture...")
    try:
        import torch
        # Import hàm build model và config của SmileNet
        from src.classifier.smile_model import build_model, SmileNetConfig
        
        # Test build SmileNet (version cơ bản)
        config_v1 = SmileNetConfig(model_name="SmileNet")
        model_v1 = build_model(config_v1)
        print("   ✅ SmileNet build successful")
        
        # Test build SmileNetV2 (có SE block)
        config_v2 = SmileNetConfig(model_name="SmileNetV2", use_se_block=True)
        model_v2 = build_model(config_v2)
        print("   ✅ SmileNetV2 build successful")
        
        # Tạo input giả để test forward pass
        dummy_input = torch.randn(2, 3, 64, 64)

        # Chạy forward
        output_v1 = model_v1(dummy_input)
        output_v2 = model_v2(dummy_input)
        
        # Kiểm tra shape output (batch_size=2, num_classes=2)
        assert output_v1.shape == (2, 2), "SmileNet output shape wrong"
        assert output_v2.shape == (2, 2), "SmileNetV2 output shape wrong"
        
        print("   ✅ Forward pass successful")
        
        # Đếm số lượng tham số của model
        params_v1 = sum(p.numel() for p in model_v1.parameters())
        params_v2 = sum(p.numel() for p in model_v2.parameters())
        print(f"   SmileNet params: {params_v1:,}")
        print(f"   SmileNetV2 params: {params_v2:,}")
        
        return True
    except Exception as e:
        # Lỗi build model hoặc forward
        print(f"   ❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_paths():
    """Test các đường dẫn data có tồn tại không."""
    print("\n🔍 Testing data paths...")
    
    try:
        import yaml

        # Load config để lấy đường dẫn data
        with open("config/train_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        image_root = Path(config['data']['image_root'])
        split_dir = Path(config['data']['split_dir'])
        
        warnings = []
        
        # Kiểm tra thư mục ảnh
        if not image_root.exists():
            warnings.append(f"Image root not found: {image_root}")
        else:
            print(f"   ✅ Image root exists: {image_root}")
        
        # Kiểm tra thư mục split CSV
        if not split_dir.exists():
            warnings.append(f"Split dir not found: {split_dir}")
        else:
            # Tìm các file CSV
            csv_files = list(split_dir.glob("*.csv"))
            if csv_files:
                print(f"   ✅ Split dir exists with {len(csv_files)} CSV files")
                for csv in csv_files:
                    print(f"      - {csv.name}")
            else:
                warnings.append(f"No CSV files in {split_dir}")
        
        # Nếu có warning thì báo
        if warnings:
            print("   ⚠️  Warnings:")
            for w in warnings:
                print(f"      - {w}")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Data path error: {e}")
        return False


def test_gpu_setup():
    """Test GPU setup và memory."""
    print("\n🔍 Testing GPU setup...")
    try:
        import torch
        
        # Nếu không có CUDA thì vẫn cho pass nhưng cảnh báo
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available - will train on CPU (slow)")
            return True
        
        # In thông tin GPU
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # Tổng bộ nhớ GPU
        device = torch.device('cuda')
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total GPU memory: {total_memory:.2f} GB")
        
        # Test cấp phát bộ nhớ GPU
        test_tensor = torch.randn(1000, 1000, device=device)
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"   Test allocation: {allocated:.2f} MB")
        
        # Giải phóng bộ nhớ
        del test_tensor
        torch.cuda.empty_cache()
        
        print("   ✅ GPU ready for training")
        return True
    except Exception as e:
        print(f"   ❌ GPU error: {e}")
        return False


def main():
    # Header
    print("="*60)
    print("🎯 Smile Detection Setup Test")
    print("="*60)
    
    # Danh sách các test cần chạy
    tests = [
        ("Imports", test_imports),
        ("Config File", test_config),
        ("Model Architecture", test_model_architecture),
        ("Data Paths", test_data_paths),
        ("GPU Setup", test_gpu_setup),
    ]
    
    results = []

    # Chạy từng test
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Tổng kết
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    # Kết luận
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Review config/train_config.yaml")
        print("  2. Run: .\\train.bat (Windows) or python train_model.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
        return 1


# Entry point
if __name__ == "__main__":
    sys.exit(main())
