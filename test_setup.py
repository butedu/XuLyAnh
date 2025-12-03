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
        import torch
        import torchvision
        import yaml
        import pandas
        import numpy
        from PIL import Image
        print("   ✅ All imports successful")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_config():
    """Test config file tồn tại và valid."""
    print("\n🔍 Testing config file...")
    config_path = Path("config/train_config.yaml")
    
    if not config_path.exists():
        print(f"   ❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['data', 'training', 'model', 'augmentation', 'settings']
        for key in required_keys:
            if key not in config:
                print(f"   ❌ Missing key in config: {key}")
                return False
        
        print("   ✅ Config file valid")
        print(f"   Model: {config['model'].get('name', 'SmileNet')}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   LR: {config['training']['learning_rate']}")
        return True
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False

def test_model_architecture():
    """Test model có thể build được không."""
    print("\n🔍 Testing model architecture...")
    try:
        import torch
        from src.classifier.smile_model import build_model, SmileNetConfig
        
        # Test SmileNet
        config_v1 = SmileNetConfig(model_name="SmileNet")
        model_v1 = build_model(config_v1)
        print("   ✅ SmileNet build successful")
        
        # Test SmileNetV2
        config_v2 = SmileNetConfig(model_name="SmileNetV2", use_se_block=True)
        model_v2 = build_model(config_v2)
        print("   ✅ SmileNetV2 build successful")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 64, 64)
        output_v1 = model_v1(dummy_input)
        output_v2 = model_v2(dummy_input)
        
        assert output_v1.shape == (2, 2), "SmileNet output shape wrong"
        assert output_v2.shape == (2, 2), "SmileNetV2 output shape wrong"
        
        print("   ✅ Forward pass successful")
        
        # Count parameters
        params_v1 = sum(p.numel() for p in model_v1.parameters())
        params_v2 = sum(p.numel() for p in model_v2.parameters())
        print(f"   SmileNet params: {params_v1:,}")
        print(f"   SmileNetV2 params: {params_v2:,}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_paths():
    """Test các đường dẫn data có tồn tại không."""
    print("\n🔍 Testing data paths...")
    
    try:
        import yaml
        with open("config/train_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        image_root = Path(config['data']['image_root'])
        split_dir = Path(config['data']['split_dir'])
        
        warnings = []
        
        if not image_root.exists():
            warnings.append(f"Image root not found: {image_root}")
        else:
            print(f"   ✅ Image root exists: {image_root}")
        
        if not split_dir.exists():
            warnings.append(f"Split dir not found: {split_dir}")
        else:
            # Check for CSV files
            csv_files = list(split_dir.glob("*.csv"))
            if csv_files:
                print(f"   ✅ Split dir exists with {len(csv_files)} CSV files")
                for csv in csv_files:
                    print(f"      - {csv.name}")
            else:
                warnings.append(f"No CSV files in {split_dir}")
        
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
        
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available - will train on CPU (slow)")
            return True
        
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # Test GPU memory
        device = torch.device('cuda')
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total GPU memory: {total_memory:.2f} GB")
        
        # Test a small tensor allocation
        test_tensor = torch.randn(1000, 1000, device=device)
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"   Test allocation: {allocated:.2f} MB")
        
        del test_tensor
        torch.cuda.empty_cache()
        
        print("   ✅ GPU ready for training")
        return True
    except Exception as e:
        print(f"   ❌ GPU error: {e}")
        return False

def main():
    print("="*60)
    print("🎯 Smile Detection Setup Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Config File", test_config),
        ("Model Architecture", test_model_architecture),
        ("Data Paths", test_data_paths),
        ("GPU Setup", test_gpu_setup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
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
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Review config/train_config.yaml")
        print("  2. Run: .\\train.bat (Windows) or python train_model.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
