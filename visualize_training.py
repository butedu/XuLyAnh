"""
Script để visualize training history và metrics.
Sử dụng: python visualize_training.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(history_path="models/training_history.json"):
    """Plot training curves từ history JSON."""
    
    if not Path(history_path).exists():
        print(f" Không tìm thấy file: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_accuracy'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    val_prec = [h['val_precision'] for h in history]
    val_recall = [h['val_recall'] for h in history]
    
    # Tìm best metrics
    best_f1_idx = val_f1.index(max(val_f1))
    best_f1 = val_f1[best_f1_idx]
    best_epoch = epochs[best_f1_idx]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - Best F1: {best_f1:.4f} @ Epoch {best_epoch}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, val_acc, 'g-', linewidth=2)
    axes[0, 1].axhline(y=max(val_acc), color='r', linestyle='--', 
                       label=f'Best: {max(val_acc):.2%}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[1, 0].plot(epochs, val_f1, 'm-', linewidth=2)
    axes[1, 0].axhline(y=best_f1, color='r', linestyle='--', 
                       label=f'Best: {best_f1:.4f}')
    axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Precision & Recall
    axes[1, 1].plot(epochs, val_prec, 'b-', label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, val_recall, 'g-', label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("models/training_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Đã lưu training curves: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Accuracy: {max(val_acc):.2%}")
    print(f"Best Precision: {max(val_prec):.2%}")
    print(f"Best Recall: {max(val_recall):.2%}")
    print(f"Final Train Loss: {train_loss[-1]:.4f}")
    print(f"Final Val Loss: {val_loss[-1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    try:
        plot_training_history()
    except Exception as e:
        print(f" Lỗi: {e}")
        import traceback
        traceback.print_exc()
