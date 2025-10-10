import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import v2
from tqdm.autonotebook import tqdm

from constants.config_const import (
    BATCH_SIZE,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
)
from constants.paths_const import DATA_PATH, LOG_PATH, TRAINED_MODEL_PATH
from dataset import PlayingCardDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_path", "-d", type=str, default=DATA_PATH)
    parser.add_argument("--image_size", "-i", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_workers", "-n", type=str, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--log_path", "-l", type=str, default=LOG_PATH)
    parser.add_argument("--save_path", "-s", type=str, default=TRAINED_MODEL_PATH)
    parser.add_argument("--checkpoint", "-sc", type=str, default=None)
    parser.add_argument("--pin_memory", "-pm", type=bool, default=True)
    parser.add_argument("--prefetch_factor", "-pf", type=int, default=2)
    parser.add_argument("--persistent_workers", "-pw", type=bool, default=True)
    parse_args = parser.parse_args()

    return parse_args


def collate_fn(batch):
    return tuple(zip(*batch))


def calculate_detailed_metrics(outputs, targets):
    """Calculate detailed metrics for object detection"""
    metrics = defaultdict(list)

    for output, target in zip(outputs, targets):
        pred_boxes = output['boxes']
        pred_scores = output['scores']
        # pred_labels = output['labels']

        gt_boxes = target['boxes']
        # gt_labels = target['labels']

        # Count predictions and ground truths
        metrics['num_predictions'].append(len(pred_boxes))
        metrics['num_ground_truths'].append(len(gt_boxes))

        # Average confidence scores
        if len(pred_scores) > 0:
            metrics['avg_confidence'].append(pred_scores.mean().item())
            metrics['max_confidence'].append(pred_scores.max().item())
            metrics['min_confidence'].append(pred_scores.min().item())
        else:
            metrics['avg_confidence'].append(0)
            metrics['max_confidence'].append(0)
            metrics['min_confidence'].append(0)

    return metrics


if __name__ == "__main__":
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_transform = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.RandomErasing(p=0.3, scale=(0.01, 0.03)),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32),
        ]
    )

    valid_transform = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32),
        ]
    )

    train_set = PlayingCardDataset(
        transform=train_transform, train=True, size=args.image_size
    )
    valid_set = PlayingCardDataset(
        transform=valid_transform, train=False, size=args.image_size
    )

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    # Model initialization
    model = fasterrcnn_mobilenet_v3_large_fpn(
        num_classes=NUM_CLASSES, trainable_backbone_layers=3
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print("Continue training at the epoch: ", start_epoch + 1)

    # Create folder save model
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # Create folder save log tensorboard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)
    writer = SummaryWriter(args.log_path)

    # Log hyperparameters
    writer.add_text(
        'Hyperparameters',
        f"""
    - Learning Rate: {args.lr}
    - Batch Size: {args.batch_size}
    - Image Size: {args.image_size}
    - Num Epochs: {args.epochs}
    - Num Workers: {args.num_workers}
    - Device: {device}
    """,
        0,
    )

    metric = MeanAveragePrecision(class_metrics=True)
    best_map = 0
    best_epoch = 0

    # Training history
    history = {'train_loss': [], 'val_map': [], 'val_map_50': [], 'val_map_75': []}

    for epoch in range(start_epoch, args.epochs):
        # Train step
        model.train()
        train_loss = []
        train_loss_classifier = []
        train_loss_box_reg = []
        train_loss_objectness = []
        train_loss_rpn_box_reg = []

        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        train_progress_bar = tqdm(train_dataloader, colour="green")
        for i, (images, targets) in enumerate(train_progress_bar):
            # Move to device
            images = [image.to(device) for image in images]
            final_targets = []
            for t in targets:
                target = {
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                }
                final_targets.append(target)

            # Forward pass
            output = model(images, final_targets)

            # Extract individual losses
            loss_value = 0
            for key, l in output.items():
                loss_value = loss_value + l

                # Track individual loss components
                if key == 'loss_classifier':
                    train_loss_classifier.append(l.item())
                elif key == 'loss_box_reg':
                    train_loss_box_reg.append(l.item())
                elif key == 'loss_objectness':
                    train_loss_objectness.append(l.item())
                elif key == 'loss_rpn_box_reg':
                    train_loss_rpn_box_reg.append(l.item())

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            train_loss.append(loss_value.item())

            # Optimize
            optimizer.step()

            # Show detailed information on train progress bar
            train_progress_bar.set_description(
                f"Train epoch {epoch + 1}/{args.epochs} | "
                f"Iter {i + 1}/{len(train_dataloader)} | "
                f"Loss: {np.mean(train_loss):.4f} | "
                f"Cls: {np.mean(train_loss_classifier) if train_loss_classifier else 0:.4f} | "
                f"Box: {np.mean(train_loss_box_reg) if train_loss_box_reg else 0:.4f} | "
                f"Obj: {np.mean(train_loss_objectness) if train_loss_objectness else 0:.4f} | "
                f"RPN: {np.mean(train_loss_rpn_box_reg) if train_loss_rpn_box_reg else 0:.4f} | "
            )

            # Save detailed training losses to tensorboard
            global_step = i + epoch * len(train_dataloader)
            writer.add_scalar('Train/Total_Loss', loss_value.item(), global_step)

            if train_loss_classifier:
                writer.add_scalar(
                    'Train/Loss_Classifier', train_loss_classifier[-1], global_step
                )
            if train_loss_box_reg:
                writer.add_scalar(
                    'Train/Loss_Box_Reg', train_loss_box_reg[-1], global_step
                )
            if train_loss_objectness:
                writer.add_scalar(
                    'Train/Loss_Objectness', train_loss_objectness[-1], global_step
                )
            if train_loss_rpn_box_reg:
                writer.add_scalar(
                    'Train/Loss_RPN_Box_Reg', train_loss_rpn_box_reg[-1], global_step
                )

        # Calculate epoch statistics
        epoch_train_loss = np.mean(train_loss)

        # Save epoch summary
        writer.add_scalar('Train/Epoch_Loss', epoch_train_loss, epoch + 1)

        if train_loss_classifier:
            writer.add_scalar(
                'Train/Epoch_Loss_Classifier', np.mean(train_loss_classifier), epoch + 1
            )
        if train_loss_box_reg:
            writer.add_scalar(
                'Train/Epoch_Loss_Box_Reg', np.mean(train_loss_box_reg), epoch + 1
            )
        if train_loss_objectness:
            writer.add_scalar(
                'Train/Epoch_Loss_Objectness', np.mean(train_loss_objectness), epoch + 1
            )
        if train_loss_rpn_box_reg:
            writer.add_scalar(
                'Train/Epoch_Loss_RPN_Box_Reg',
                np.mean(train_loss_rpn_box_reg),
                epoch + 1,
            )

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  - Total Loss: {epoch_train_loss:.4f}")
        print(f"  - Learning Rate: {current_lr:.6f}")

        # Validation step
        model.eval()
        val_progress_bar = tqdm(valid_dataloader, colour="yellow")

        # Validation metrics
        val_predictions_count = []
        val_gt_count = []
        val_confidence_scores = []

        for i, (images, targets) in enumerate(val_progress_bar):
            # Move tensors to the configured device
            images = list(image.to(device) for image in images)
            targets_list = []
            for t in targets:
                target = {
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                }
                targets_list.append(target)

            # Model prediction
            with torch.no_grad():
                outputs = model(images)

            # Calculate detailed metrics
            detailed_metrics = calculate_detailed_metrics(outputs, targets_list)
            val_predictions_count.extend(detailed_metrics['num_predictions'])
            val_gt_count.extend(detailed_metrics['num_ground_truths'])
            val_confidence_scores.extend(detailed_metrics['avg_confidence'])

            boxes = []
            labels = []
            scores = []
            # Get prediction
            for prediction in outputs:
                boxes.append(prediction["boxes"])
                labels.append(prediction["labels"])
                scores.append(prediction["scores"])

            preds = []
            for b, l, s in zip(boxes, labels, scores):
                preds.append(dict(boxes=b, scores=s, labels=l))

            metric.update(preds, targets_list)

            # Show detailed information on valid progress bar
            val_progress_bar.set_description(
                f"Valid epoch {epoch + 1}/{args.epochs} | "
                f"Iter {i + 1}/{len(valid_dataloader)} | "
                f"Avg Preds: {np.mean(val_predictions_count):.1f} | "
                f"Avg Ground Truths: {np.mean(val_gt_count):.1f} | "
                f"Avg Conf: {np.mean(val_confidence_scores):.3f}"
            )

        # Calculate mean average precision
        map_dict = metric.compute()
        metric.reset()

        # Save all mAP metrics to tensorboard
        writer.add_scalar("Valid/mAP", map_dict["map"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP_50", map_dict["map_50"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP_75", map_dict["map_75"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP_small", map_dict["map_small"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP_medium", map_dict["map_medium"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP_large", map_dict["map_large"].item(), epoch + 1)
        writer.add_scalar("Valid/mAR_1", map_dict["mar_1"].item(), epoch + 1)
        writer.add_scalar("Valid/mAR_10", map_dict["mar_10"].item(), epoch + 1)
        writer.add_scalar("Valid/mAR_100", map_dict["mar_100"].item(), epoch + 1)

        # Additional validation metrics
        writer.add_scalar(
            "Valid/Avg_Predictions_Per_Image", np.mean(val_predictions_count), epoch + 1
        )
        writer.add_scalar(
            "Valid/Avg_Ground_Truths_Per_Image", np.mean(val_gt_count), epoch + 1
        )
        writer.add_scalar(
            "Valid/Avg_Confidence_Score", np.mean(val_confidence_scores), epoch + 1
        )

        # Per-class metrics (if available)
        if 'map_per_class' in map_dict:
            for class_idx, class_map in enumerate(map_dict['map_per_class']):
                if not torch.isnan(class_map):
                    writer.add_scalar(
                        f"Valid/mAP_Class_{class_idx}", class_map.item(), epoch + 1
                    )

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_map'].append(map_dict["map"].item())
        history['val_map_50'].append(map_dict["map_50"].item())
        history['val_map_75'].append(map_dict["map_75"].item())

        print(f"\nEpoch {epoch + 1} Validation Summary:")
        print(f"  - mAP: {map_dict['map'].item():.4f}")
        print(f"  - mAP@50: {map_dict['map_50'].item():.4f}")
        print(f"  - mAP@75: {map_dict['map_75'].item():.4f}")
        print(f"  - mAP (small): {map_dict['map_small'].item():.4f}")
        print(f"  - mAP (medium): {map_dict['map_medium'].item():.4f}")
        print(f"  - mAP (large): {map_dict['map_large'].item():.4f}")
        print(f"  - mAR@1: {map_dict['mar_1'].item():.4f}")
        print(f"  - mAR@10: {map_dict['mar_10'].item():.4f}")
        print(f"  - mAR@100: {map_dict['mar_100'].item():.4f}")
        print(f"  - Avg Predictions/Image: {np.mean(val_predictions_count):.1f}")
        print(f"  - Avg Ground Truths/Image: {np.mean(val_gt_count):.1f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))

        if map_dict["map"].item() > best_map:
            best_map = map_dict["map"].item()
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            print(f"New best model saved! Best mAP: {best_map:.4f}")
        else:
            print(f"Current best mAP: {best_map:.4f} (Epoch {best_epoch + 1})")

        # Early Stopping Check
        if epoch - best_epoch >= 3:
            print(f"\nEarly stopping activated at epoch {epoch + 1}")
            break

    # Training completed
    print(f"Best mAP: {best_map:.4f} at epoch {best_epoch + 1}")

    # Close tensorboard writer
    writer.close()
