import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import math
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from model.yolov4 import YOLOv4
from voc import VOCDetection, CollateFunc
from augmentation import SSDAugmentation, SSDBaseTransform
from loss import build_criterion
import config
from eval import predict_yolov4_batch, build_gts_from_targets, evaluate_map


@torch.no_grad()
def validate(model, val_loader, device):
    """
    Evaluate model on VOC2007 test set and compute mAP@0.5
    """
    model.eval()

    all_preds = []
    all_gts = []

    for images, targets in tqdm(val_loader, desc="Validation"):
        images = images.to(device)

        # batch inference and coordinate decoding
        preds = predict_yolov4_batch(
            model,
            images,
            conf_thresh=0.001,
            nms_thresh=0.5
        )
        # convert ground truth targets to evaluation format
        gts = build_gts_from_targets(targets, img_size=config.img_size)

        all_preds.extend(preds)
        all_gts.extend(gts)

    # calculate Mean Average Precision (mAP@0.5)
    mAP, ap_per_class = evaluate_map(
        all_preds,
        all_gts,
        num_classes=config.num_classes,  # VOC=20
        iou_thresh=0.5,
        use_07_metric=False
    )

    return mAP, ap_per_class


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def build_train_loader(train_dataset):
    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=CollateFunc(),
        pin_memory=True,
        drop_last=True
    )


def train():
    # set_seed(config.seed)

    if config.device == 'cuda':
        device = torch.device('cuda')
        gpu = torch.cuda.get_device_name(device)
        tqdm.write(f"💻 Device: {gpu}")
    else:
        tqdm.write("💻 Device: CPU")

    # -------------------- paths --------------------
    save_dir = Path(config.save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "latest_checkpoint.pth"
    history_path = save_dir / "history.pkl"
    best_path = save_dir / "yolov4_voc_best.pth"

    # -------------------- Dataset --------------------
    train_transform = SSDAugmentation(img_size=config.img_size)
    val_transform = SSDBaseTransform(img_size=config.img_size)

    train_dataset = VOCDetection(
        root=config.root,
        image_sets=config.train_sets,
        transform=train_transform,
        is_train=True
    )

    val_dataset = VOCDetection(
        root=config.root,
        image_sets=config.val_sets,
        transform=val_transform,
        is_train=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=CollateFunc(),
        pin_memory=True
    )

    # -------------------- Model & Loss --------------------
    model = YOLOv4(
        device=device,
        num_classes=config.num_classes,
        anchor_size=config.anchor_size,
        trainable=True,
        conf_thresh=config.conf_thresh,
        nms_thresh=config.nms_thresh,
        topk=config.topk,
        model_name='cspdarknet53',
        pretrained=config.pretrained
    ).to(device)

    criterion = build_criterion(
        device=device,
        num_classes=config.num_classes,
        obj_w=config.obj_weight,
        cls_w=config.cls_weight,
        box_w=config.box_weight
    )

    # -------------------- Optimizer --------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # -------------------- Scheduler --------------------
    def lr_lambda(epoch):
        if epoch < config.wp_epoch:
            # linear warmup
            return 0.1 + 0.9 * ((epoch + 1) / config.wp_epoch)
        else:
            # cosine decay
            t = epoch - config.wp_epoch
            T = max(1, config.max_epoch - config.wp_epoch - 1)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))
            return 0.01 + (1.0 - 0.01) * cos_factor

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------- AMP --------------------
    scaler = GradScaler('cuda') if config.use_amp and config.device == 'cuda' else None

    # -------------------- Resume --------------------
    start_epoch = 0
    best_map = 0.0
    history = {'train_loss': [], 'mAP': []}

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)

        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            tqdm.write(f"Loaded history: {len(history['train_loss'])} epochs recorded")

        tqdm.write(f"Resumed from epoch {start_epoch} (best mAP so far: {best_map * 100:.2f}%)")
    else:
        tqdm.write("No checkpoint found, start training from scratch")

    tqdm.write("🚀 Start training...")

    # -------------------- Multi-scale --------------------
    multi_scale_sizes = getattr(config, 'multi_scale_sizes', [448])

    current_size = random.choice(getattr(config, 'multi_scale_sizes', [448]))
    train_transform.set_img_size(current_size)

    train_loader = build_train_loader(train_dataset)

    # -------------------- Train --------------------
    for epoch in range(start_epoch, config.max_epoch):
        current_seed = config.seed + epoch 
        set_seed(current_seed)
        
        new_size = random.choice(multi_scale_sizes)
        train_transform.set_img_size(new_size)
        train_loader = build_train_loader(train_dataset)
        tqdm.write(f"multi-scale: img_size = {new_size}")

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epoch} | Train")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            optimizer.zero_grad()

            if config.use_amp and scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets, epoch)
                    loss = loss_dict['losses']

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, targets, epoch)
                loss = loss_dict['losses']
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'lr': f"{current_lr:.8f}"
            })

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # -------------------- Validation --------------------
        mAP, ap_per_class = validate(model, val_loader, device)
        history['mAP'].append(mAP)

        tqdm.write(f'avg_loss: {avg_loss:.4f} | mAP@0.5 = {mAP * 100:.2f}%')

        # best
        if mAP > best_map:
            best_map = mAP
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'mAP': best_map,
            }, best_path)
            tqdm.write(f'New best model saved: mAP = {mAP * 100:.2f}%')
        tqdm.write('')

        # lateset
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_map': best_map,
        }
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

    print('\n🎉 Training complete!')
    print(f'Best model saved with mAP@0.5: {best_map * 100:.2f}%')

    plt.figure(figsize=(15, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # -------------------- Draw Curves --------------------
    ## loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('YOLOv4 Training Loss')
    plt.ylim(0, 15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    ## mAP curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x * 100 for x in history['mAP']], label='Val mAP@0.5', color='#d62728', marker='o', markersize=4)
    plt.xlabel('Epochs')
    plt.ylabel('mAP (%)')
    plt.title('Validation mAP @ 0.5')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    ## save and show
    plt.tight_layout()
    plot_path = save_dir / 'yolov4_training_metrics.png'
    plt.savefig(plot_path, dpi=150)
    tqdm.write(f"📊 Training metrics plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    train()
