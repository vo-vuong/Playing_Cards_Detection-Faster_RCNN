import os
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import cv2
import numpy as np
from constants.paths_const import DATA_PATH, ANNOTATIONS_ORG_FILE
from constants.config_const import CLASSES


class PlayingCardDataset(Dataset):
    def __init__(self, root=DATA_PATH, train=True, transform=None, size=640):
        self.images = []
        self.annotations = []
        self.size = size
        self.transform = transform
        self.classes = CLASSES

        self.data_path = root
        if train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "valid")

        # Read annotations file
        annotations_file = os.path.join(self.data_path, ANNOTATIONS_ORG_FILE)
        with open(annotations_file, "r") as file:
            data_annotation = json.load(file)
        self.images = data_annotation["images"]
        self.annotations = data_annotation["annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        current_image = self.images[idx]
        image_path = os.path.join(self.data_path, current_image["file_name"])
        image = Image.open(image_path).convert("RGB")

        # get annotations of this item
        annotations = []
        for ann in self.annotations:
            if ann["image_id"] == self.images[idx]["id"]:
                annotations.append(ann)

        boxes = []
        labels = []
        ori_width = int(current_image["width"])
        ori_height = int(current_image["height"])
        for annotation in annotations:
            bbox = annotation["bbox"]
            x_min = int(bbox[0]) / ori_width * self.size
            y_min = int(bbox[1]) / ori_height * self.size
            x_max = int(bbox[2] + bbox[0]) / ori_width * self.size
            y_max = int(bbox[3] + bbox[1]) / ori_height * self.size
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation["category_id"])
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        targets = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, targets


if __name__ == "__main__":
    train_set = PlayingCardDataset(train=False)

    image, label = train_set.__getitem__(1)
    cv2_image = np.array(image)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", cv2_image)
    cv2.waitKey(0)
