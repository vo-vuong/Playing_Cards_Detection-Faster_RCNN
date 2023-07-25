import os
import shutil
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter
from dataset import PlayingCardDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from constants.paths_const import DATA_PATH, TRAINED_MODEL_PATH, LOG_PATH
from constants.config_const import (
    BATCH_SIZE,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    IMAGE_SIZE,
    NUM_EPOCHS,
    NUM_CLASSES,
)


def get_args():
    parser = argparse.ArgumentParser(description="Train a EmotionCNN model")
    parser.add_argument("--data_path", "-d", type=str, default=DATA_PATH)
    parser.add_argument("--image_size", "-i", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_workers", "-n", type=str, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--log_path", "-l", type=str, default=LOG_PATH)
    parser.add_argument("--save_path", "-s", type=str, default=TRAINED_MODEL_PATH)
    parser.add_argument("--checkpoint", "-sc", type=str, default=None)
    args = parser.parse_args()

    return args


def collate_fn(batch):
    images = []
    targets = []

    for i, t in batch:
        images.append(i)
        targets.append(t)

    return images, targets


if __name__ == "__main__":
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            ToTensor(),
            ColorJitter(brightness=0.125, contrast=0.1, saturation=0.1, hue=0.05),
            Normalize(mean=[0.5531, 0.5128, 0.4813], std=[0.2496, 0.2552, 0.2591]),
        ]
    )

    valid_transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            ToTensor(),
            Normalize(mean=[0.5531, 0.5128, 0.4813], std=[0.2496, 0.2552, 0.2591]),
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
    )

    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model initialization
    model = fasterrcnn_mobilenet_v3_large_fpn(
        num_classes=NUM_CLASSES, trainable_backbone_layers=6
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print("Continue training at the epoch: ", start_epoch + 1)
    else:
        start_epoch = 0

    # Create folder save model
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # Create folder save log tensorboard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)
    writer = SummaryWriter(args.log_path)

    metric = MeanAveragePrecision(class_metrics=True)
    best_map = 0
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # Train step
        model.train()
        train_loss = []

        train_progress_bar = tqdm(train_dataloader, colour="green")
        for i, (images, targets) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            final_targets = []
            for t in targets:
                target = {}
                target["boxes"] = t["boxes"].to(device)
                target["labels"] = t["labels"].to(device)
                final_targets.append(target)

            # Forward pass
            output = model(images, final_targets)
            loss_value = 0
            for l in output.values():
                loss_value = loss_value + l

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            train_loss.append(loss_value.item())

            # Optimize
            optimizer.step()

            # Show information on train progress bar
            train_progress_bar.set_description(
                "Train epoch {}. Iteration {}/{} Loss {:0.4f}".format(
                    epoch + 1, i + 1, len(train_dataloader), np.mean(train_loss)
                )
            )
            # Save training loss to tensorboard
            writer.add_scalar(
                "Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader)
            )

        # Validation step
        model.eval()
        val_progress_bar = tqdm(valid_dataloader, colour="yellow")
        for i, (images, targets) in enumerate(val_progress_bar):
            # Move tensors to the configured device
            images = list(image.to(device) for image in images)
            targets_list = []
            for t in targets:
                target = {}
                target["boxes"] = t["boxes"].to(device)
                target["labels"] = t["labels"].to(device)
                targets_list.append(target)

            # Model prediction
            with torch.no_grad():
                outputs = model(images)

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

            # Show information on valid progress bar
            val_progress_bar.set_description(
                "Valid epoch {}. Iteration {}/{}".format(
                    epoch + 1, i + 1, len(valid_dataloader)
                )
            )

        # Calculate mean average precision
        map_dict = metric.compute()
        # Save mAP to tensorboard
        writer.add_scalar("Valid/mAP", map_dict["map"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP50", map_dict["map_50"].item(), epoch + 1)
        writer.add_scalar("Valid/mAP75", map_dict["map_75"].item(), epoch + 1)

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

        if epoch - best_epoch == 3:  # Early Stopping
            print("Activate early stopping at epoch: ", epoch + 1)
            break
