import torch
from src.classifier.smile_model import SmileNetConfig, build_model

def test_smilenet():
    print("Testing SmileNet (baseline)...")
    
    # Tạo config cho model SmileNet (bản baseline)
    config = SmileNetConfig(model_name="SmileNet")
    
    # Xây dựng model từ config
    model = build_model(config)
    
    # Tạo input giả để test forward pass (batch size = 2, ảnh RGB 64x64)
    dummy_input = torch.randn(2, 3, 64, 64)
    
    # Chạy mô hình để lấy output
    output = model(dummy_input)
    
    # Kiểm tra kích thước output đúng dạng (batch_size, num_classes = 2)
    assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
    
    # Đếm số lượng parameters của model
    params = sum(p.numel() for p in model.parameters())
    print(f" SmileNet OK - {params:,} parameters")
    return True


def test_smilenetv2():
    print("Testing SmileNetV2 (improved)...")
    
    # Tạo config cho SmileNetV2, bật SE block
    config = SmileNetConfig(model_name="SmileNetV2", use_se_block=True)
    
    # Xây dựng model SmileNetV2
    model = build_model(config)
    
    # Input giả giống như test trên
    dummy_input = torch.randn(2, 3, 64, 64)
    
    # Forward pass
    output = model(dummy_input)
    
    # Kiểm tra output có đúng kích thước
    assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
    
    # In số lượng parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"SmileNetV2 OK - {params:,} parameters")
    return True


if __name__ == "__main__":
    try:
        # Chạy thử hai bài test
        test_smilenet()
        test_smilenetv2()
        
        # Nếu không lỗi => thành công
        print("\n All model tests passed!")
    except Exception as e:
        # In lỗi nếu có test thất bại
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
