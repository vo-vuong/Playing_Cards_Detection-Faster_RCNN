import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxFormat

from constants.config_const import CLASSES
from constants.paths_const import ANNOTATIONS_ORG_FILE, DATA_PATH


class PlayingCardDataset(Dataset):
    def __init__(self, root=DATA_PATH, train=True, transform=None, size=640):
        self.transform = transform
        self.size = size
        self.classes = CLASSES

        self.data_path = root
        if train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "valid")

        # Read annotations file
        annotations_file = os.path.join(self.data_path, ANNOTATIONS_ORG_FILE)
        with open(annotations_file, "r", encoding="utf-8") as file:
            data_annotation = json.load(file)

        self.images = data_annotation["images"]
        self.image_to_annotations = self._build_annotation_index(
            data_annotation["annotations"]
        )

    def _build_annotation_index(self, annotations):
        index = defaultdict(list)
        for ann in annotations:
            index[ann["image_id"]].append(ann)
        return index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        current_image = self.images[idx]
        image_path = os.path.join(self.data_path, current_image["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Store original image size for BoundingBoxes format
        img_width, img_height = image.size

        # get annotations of this item
        annotations = self.image_to_annotations.get(current_image["id"], [])

        boxes = []
        labels = []

        for annotation in annotations:
            bbox = annotation["bbox"]
            # Normalize coordinates to target size
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]

            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(annotation["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Convert to TVTensors format - this is the key change!
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=BoundingBoxFormat.XYXY,
            canvas_size=(img_height, img_width),
        )

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


if __name__ == "__main__":
    train_set = PlayingCardDataset(train=False)

    img, targets = train_set[1]
    cv2_image = np.array(img)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes for visualization
    for box in targets["boxes"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("image", cv2_image)
    cv2.waitKey(0)
