import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2


# --- Global batch setting ---
_batch_size = 8  # default


def set_batch(batch=8):
    global _batch_size
    _batch_size = batch
    

class AlbumentationsDigitCocoDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None):
        self.coco = COCO(ann_path)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())


    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        image_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        bboxes = []
        category_ids = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            bboxes.append([x_min, y_min, x_max, y_max])
            category_ids.append(ann['category_id'])

        if self.transform:
            transformed = self.transform(
                image=img_np,
                bboxes=bboxes,
                category_ids=category_ids
            )
            img_tensor = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']
        else:
            raise ValueError("Albumentations transform is required")

        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(category_ids, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return img_tensor, target


    def __len__(self):
        return len(self.ids)
   
class TestDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ])
        self.image_ids = [
            int(os.path.splitext(os.path.basename(p))[0]) for p in self.img_paths
        ]


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size  # 原圖尺寸
        if self.transforms:
            image = self.transforms(image)
        return image, self.image_ids[idx], (width, height)  # ⬅️ 傳回原圖尺寸


# --- collate_fn ---
def collate_fn(batch):
    return tuple(zip(*batch))


def collate_fn_test(batch):
    images, image_ids, original_sizes = zip(*batch)
    return list(images), list(image_ids), list(original_sizes)


# --- Transforms ---
transform = T.Compose([T.ToTensor()])


# --- Paths ---
TRAIN_IMG_DIR = "./data/train"
VAL_IMG_DIR = "./data/valid"
TEST_IMG_DIR = "./data/test"


TRAIN_ANN_PATH = "./data/train.json"
VAL_ANN_PATH = "./data/valid.json"


IMG_SIZE = 256  # 建議加個統一變數


def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15),
            shear={"x": (-5, 5), "y": (-5, 5)},
            fit_output=False,
            keep_ratio=True,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            rotate_method="largest_box",
            balanced_scale=True,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.3),
        A.RandomFog(p=0.4),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.OneOf([
            A.ToGray(p=1.0),
            A.ChannelDropout(p=1.0)
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(3, 6),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            fill="inpaint_ns",
            p=1.0
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['category_ids'], 
        min_visibility=0.8))


def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))



val_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# --- Loader functions ---
def get_train_loader():
    dataset = AlbumentationsDigitCocoDataset(
        TRAIN_IMG_DIR, TRAIN_ANN_PATH, transform=get_train_transform())
    return DataLoader(dataset, batch_size=_batch_size, shuffle=True, collate_fn=collate_fn)


def get_val_loader():
    dataset = AlbumentationsDigitCocoDataset(
        VAL_IMG_DIR, VAL_ANN_PATH, transform=get_val_transform())
    return DataLoader(dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_fn)


def get_test_loader():
    dataset = TestDataset(TEST_IMG_DIR, transforms=val_transform)
    loader = DataLoader(dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_fn_test)
    return loader


if __name__ == '__main__':
    # Load the JSON file
    with open('./data/train.json', 'r') as file:
        data = json.load(file)

    # Print all keys (top-level)
    for key in data.keys():
        print(f"{key} cols:", end = ' ')
        for keyy in data[key][0].keys():
            print(keyy, end = ' ')
        print()
       
    train_loader = get_train_loader()
    val_loader   = get_val_loader()
    test_loader  = get_test_loader()