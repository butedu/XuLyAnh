import torch
from src.classifier.smile_model import SmileNetConfig, build_model

def test_smilenet():
    print("Testing SmileNet (baseline)...")
    config = SmileNetConfig(model_name="SmileNet")
    model = build_model(config)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)
    
    assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
    
    params = sum(p.numel() for p in model.parameters())
    print(f" SmileNet OK - {params:,} parameters")
    return True

def test_smilenetv2():
    print("Testing SmileNetV2 (improved)...")
    config = SmileNetConfig(model_name="SmileNetV2", use_se_block=True)
    model = build_model(config)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)
    
    assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SmileNetV2 OK - {params:,} parameters")
    return True

if __name__ == "__main__":
    try:
        test_smilenet()
        test_smilenetv2()
        print("\n All model tests passed!")
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
