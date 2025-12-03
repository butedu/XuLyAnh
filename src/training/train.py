"""Huấn luyện bộ phân loại nụ cười."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from src.classifier.smile_model import IMAGE_SIZE, build_model, SmileNetConfig
from src.training.data import LoaderConfig, build_loaders
from src.training.utils import classification_report, save_checkpoint, set_seed


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """Xây dựng data augmentation từ config với augmentation mạnh hơn."""
    aug_cfg = config.get('augmentation', {})
    image_size = tuple(config['data']['image_size'])
    
    # Normalization
    norm_cfg = aug_cfg.get('normalize', {})
    normalize = transforms.Normalize(
        mean=norm_cfg.get('mean', [0.485, 0.456, 0.406]),
        std=norm_cfg.get('std', [0.229, 0.224, 0.225])
    )
    
    # Training augmentation - mạnh hơn nhiều
    train_transforms = [
        transforms.Resize(image_size),
    ]
    
    # Horizontal flip
    if aug_cfg.get('horizontal_flip', True):
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    # Random rotation
    rotation_deg = aug_cfg.get('random_rotation', 0)
    if rotation_deg > 0:
        train_transforms.append(transforms.RandomRotation(rotation_deg))
    
    # Color jitter
    color_cfg = aug_cfg.get('color_jitter', {})
    if color_cfg:
        train_transforms.append(transforms.ColorJitter(
            brightness=color_cfg.get('brightness', 0.3),
            contrast=color_cfg.get('contrast', 0.3),
            saturation=color_cfg.get('saturation', 0.2),
            hue=color_cfg.get('hue', 0.1)
        ))
    
    # Gaussian blur
    blur_cfg = aug_cfg.get('gaussian_blur', {})
    if blur_cfg.get('probability', 0) > 0:
        train_transforms.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=blur_cfg.get('kernel_size', 5))
        ], p=blur_cfg['probability']))
    
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)
    
    # Random erasing (cutout) - sau khi ToTensor
    erase_cfg = aug_cfg.get('random_erasing', {})
    if erase_cfg.get('probability', 0) > 0:
        train_transforms.append(transforms.RandomErasing(
            p=erase_cfg['probability'],
            scale=tuple(erase_cfg.get('scale', [0.02, 0.15])),
            ratio=tuple(erase_cfg.get('ratio', [0.3, 3.3])),
            value='random'
        ))
    
    # Validation/test augmentation - chỉ resize và normalize
    eval_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ]
    
    return {
        "train": transforms.Compose(train_transforms),
        "eval": transforms.Compose(eval_transforms)
    }


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Xây dựng optimizer từ config."""
    train_cfg = config['training']
    opt_cfg = train_cfg.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'adamw').lower()
    lr = train_cfg['learning_rate']
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    
    if opt_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                     betas=tuple(opt_cfg.get('betas', [0.9, 0.999])))
    elif opt_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                    betas=tuple(opt_cfg.get('betas', [0.9, 0.999])))
    elif opt_type == 'sgd':
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                   momentum=opt_cfg.get('momentum', 0.9), nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], steps_per_epoch: int):
    """Xây dựng learning rate scheduler với warmup."""
    train_cfg = config['training']
    sched_cfg = train_cfg.get('scheduler', {})
    sched_type = sched_cfg.get('type', 'cosine').lower()
    
    warmup_epochs = sched_cfg.get('warmup_epochs', 0)
    total_epochs = train_cfg['epochs']
    
    if sched_type == 'cosine':
        min_lr = sched_cfg.get('min_lr', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
    elif sched_type == 'step':
        step_size = sched_cfg.get('step_size', 10)
        gamma = sched_cfg.get('gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    return scheduler, warmup_epochs


def warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linear warmup của learning rate."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    grad_clip: float | None = None
) -> float:
    """Training loop với mixed precision và gradient clipping."""
    model.train()
    loss_sum = 0.0
    
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if use_amp and scaler is not None:
            with autocast():  # PyTorch 2.x autocast cho CUDA
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        loss_sum += loss.item() * images.size(0)
    
    return loss_sum / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    metrics = classification_report(np.array(targets), np.array(preds))
    return loss_sum / len(loader.dataset), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện CNN phân loại nụ cười")
    parser.add_argument("--config", type=Path, default=Path("config/train_config.yaml"),
                       help="Đường dẫn file YAML config")
    # Override options (optional)
    parser.add_argument("--image-root", type=Path, help="Override image root từ config")
    parser.add_argument("--split-dir", type=Path, help="Override split dir từ config")
    parser.add_argument("--output-dir", type=Path, help="Override output dir từ config")
    parser.add_argument("--epochs", type=int, help="Override số epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--resume", type=Path, help="Resume từ checkpoint")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    # Load config từ YAML
    config = load_config(args.config)
    
    # Override config với command line arguments nếu có
    if args.image_root:
        config['data']['image_root'] = str(args.image_root)
    if args.split_dir:
        config['data']['split_dir'] = str(args.split_dir)
    if args.output_dir:
        config['data']['output_dir'] = str(args.output_dir)
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['settings']['device'] = args.device
    
    # Setup
    set_seed(config['settings']['seed'])
    
    # Device setup
    device_str = config['settings']['device']
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    if device.type == 'cuda':
        print(f"🚀 Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  Đang chạy trên CPU (chậm hơn GPU)")
    
    # Build transforms với augmentation mạnh
    tfs = build_transforms(config)
    
    # Build data loaders
    train_cfg = config['training']
    loaders = build_loaders(
        image_root=config['data']['image_root'],
        split_dir=config['data']['split_dir'],
        transform=tfs["train"],
        config=LoaderConfig(
            batch_size=train_cfg['batch_size'],
            num_workers=config['settings']['num_workers'],
            pin_memory=config['settings']['pin_memory']
        ),
    )
    
    for name in ("train", "val"):
        if name not in loaders:
            raise ValueError(f"Thiếu tập {name}. Hãy tạo file {name}.csv trong {config['data']['split_dir']}")
    
    # Apply eval transforms cho val/test
    eval_tf = tfs["eval"]
    for split in ("val", "test"):
        if split in loaders:
            dataset = loaders[split].dataset
            if hasattr(dataset, "transform"):
                setattr(dataset, "transform", eval_tf)
    
    # Build model
    model_cfg = config['model']
    model_config = SmileNetConfig(
        dropout=model_cfg['dropout'],
        model_name=model_cfg.get('name', 'SmileNet'),
        use_se_block=model_cfg.get('use_se_block', True)
    )
    model = build_model(model_config).to(device)
    
    print(f"\n📊 Mô hình: {model_cfg.get('name', 'SmileNet')}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Tổng số parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}\n")
    
    # Build optimizer và scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler, warmup_epochs = build_scheduler(optimizer, config, len(loaders['train']))
    
    # Mixed precision setup
    use_amp = config['settings'].get('use_amp', False) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("⚡ Mixed Precision Training: ENABLED")
    
    # Gradient clipping
    grad_clip_cfg = config['settings'].get('grad_clip', {})
    grad_clip = grad_clip_cfg.get('max_norm') if grad_clip_cfg.get('enabled', False) else None
    
    # Early stopping setup
    early_stop_cfg = config['settings'].get('early_stopping', {})
    early_stopping = early_stop_cfg.get('enabled', False)
    patience = early_stop_cfg.get('patience', 15)
    min_delta = early_stop_cfg.get('min_delta', 0.001)
    patience_counter = 0
    
    # Output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume từ checkpoint nếu có
    start_epoch = 1
    best_f1 = 0.0
    if args.resume and args.resume.exists():
        print(f"📂 Loading checkpoint từ {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('best_metric', 0.0)
        print(f"   Resuming từ epoch {start_epoch}, best F1 = {best_f1:.4f}")
    
    history: list[Dict[str, float]] = []
    base_lr = config['training']['learning_rate']

    print(f"\n🎯 Bắt đầu training {config['training']['epochs']} epochs...\n")
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Warmup learning rate
        if epoch <= warmup_epochs:
            warmup_lr(optimizer, epoch - 1, warmup_epochs, base_lr)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🔥 Warmup LR: {current_lr:.6f}")
        
        # Training
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            use_amp=use_amp, scaler=scaler, grad_clip=grad_clip
        )
        
        # Validation
        val_loss, val_metrics = evaluate(model, loaders["val"], criterion, device)
        
        # Step scheduler
        if scheduler is not None and epoch > warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.get('f1', 0.0))
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_history)
        
        # Print metrics
        print(f"\n📈 Metrics:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val Acc:    {val_metrics['accuracy']:.2%}")
        print(f"   Val F1:     {val_metrics['f1']:.4f}")
        print(f"   Val Prec:   {val_metrics['precision']:.2%}")
        print(f"   Val Recall: {val_metrics['recall']:.2%}")
        print(f"   LR:         {current_lr:.6f}")
        
        current_f1 = val_metrics.get("f1", 0.0)
        
        # Save best model
        if current_f1 > best_f1 + min_delta:
            improvement = current_f1 - best_f1
            best_f1 = current_f1
            patience_counter = 0
            
            best_path = output_dir / "smile_cnn_best.pth"
            torch.save(model.state_dict(), best_path)
            
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_f1,
                path=output_dir / "checkpoint.pt",
            )
            print(f"\n💾 Lưu mô hình tốt nhất: F1 = {best_f1:.4f} (+{improvement:.4f})")
        else:
            patience_counter += 1
            if early_stopping:
                print(f"   No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if early_stopping and patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered! No improvement for {patience} epochs.")
            break
        
        # Save periodic checkpoint
        save_freq = config['settings'].get('save_frequency', 5)
        if epoch % save_freq == 0:
            periodic_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(epoch, model, optimizer, scheduler, best_f1, periodic_path)
            print(f"💾 Checkpoint định kỳ: {periodic_path.name}")
        
        print()  # Newline for readability
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Hoàn tất huấn luyện!")
    print(f"   Best F1 Score: {best_f1:.4f}")
    print(f"   Model saved: {output_dir / 'smile_cnn_best.pth'}")
    print(f"   History saved: {history_path}")
    print(f"{'='*60}\n")
    
    # Test evaluation nếu có
    if config['validation'].get('test_after_training', False) and 'test' in loaders:
        print("🧪 Đánh giá trên tập Test...")
        model.load_state_dict(torch.load(output_dir / "smile_cnn_best.pth", map_location=device))
        test_loss, test_metrics = evaluate(model, loaders["test"], criterion, device)
        print(f"   Test Loss:     {test_loss:.4f}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.2%}")
        print(f"   Test F1:       {test_metrics['f1']:.4f}")
        print(f"   Test Precision:{test_metrics['precision']:.2%}")
        print(f"   Test Recall:   {test_metrics['recall']:.2%}\n")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
