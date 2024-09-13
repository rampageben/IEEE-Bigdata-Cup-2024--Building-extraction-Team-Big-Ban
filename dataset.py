
from torchvision.datasets import CocoDetection
import torch
import numpy as np
import os
from torchvision.transforms import ToTensor
from transforms import  RandomHorizontalFlip, MosaicWithBBoxes, RandomVerticalFlip


class CustomCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, model=None, transform=None, target_transform=None):
        super(CustomCocoDetection, self).__init__(img_folder, ann_file)
        self.transform = transform
        self.target_transform = target_transform
        self.model = model
        self.random_horizontal_flip = RandomHorizontalFlip(p=0.5)
        self.random_vertical_flip = RandomVerticalFlip(p=0.5)
        self.mosaic = MosaicWithBBoxes(img_size=256)
        self.mosaic_prob = 0.5

    def __getitem__(self, idx):
        img, ann = super(CustomCocoDetection, self).__getitem__(idx)  
        imgs, anns = [img], [ann]

        mosaic = True if self.model == 'train' and np.random.rand() < self.mosaic_prob else False
        while True: 
            if mosaic:
                for i in range(3): 
                    img, ann = super(CustomCocoDetection, self).__getitem__(np.random.randint(0, len(self.ids)))
                    imgs.append(img)
                    anns.append(ann)
            
            all_boxes = []
            all_labels = []
            all_masks = []
            idx_list = [0, 1, 2, 3] if mosaic else [0]
            for imgidx, img, ann in zip(idx_list, imgs, anns):
                num_objs = len(ann)

                boxes = []
                labels = []
                masks = []

                for i in range(num_objs):
                    xmin = float(ann[i]['bbox'][0])
                    ymin = float(ann[i]['bbox'][1])
                    xmax = float(float(ann[i]['bbox'][0]) + float(ann[i]['bbox'][2]))
                    ymax = float(float(ann[i]['bbox'][1]) + float(ann[i]['bbox'][3]))
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(ann[i]['category_id'] + 1)
                    masks.append(self.coco.annToMask(ann[i]))

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks = np.stack(masks, axis=0) if masks else np.empty((0, img.height, img.width), dtype=np.uint8)
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                image_id = torch.tensor([idx])

                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["masks"] = masks
                target["image_id"] = image_id

                if self.model == 'train':
                    target = self.remove_duplicates(target)
                    img, target = self.random_horizontal_flip(img, target)
                    img, target = self.random_vertical_flip(img, target)

                imgs[imgidx] = img
                
                all_boxes.append(target['boxes'])
                all_labels.append(target['labels'])
                all_masks.append(target['masks'])
            
            if mosaic:
                img, target = self.mosaic(imgs, all_boxes, all_labels, all_masks)
                if target["boxes"].shape[0] > 0:
                    break
                continue
            else:
                img = imgs[0]
                target = {'boxes': all_boxes[0], 'labels': all_labels[0], 'masks': all_masks[0]}
                break

        img = ToTensor()(img)
        return img, target
    def remove_duplicates(self, target):
        max_aspect_ratio = 5
        min_bbox_area = 10
        min_mask_area = 10
        boxes = target['boxes']
        masks = target['masks']
        labels = target['labels']

        unique_boxes = []
        unique_masks = []
        unique_labels = []

        for i in range(len(masks)):
            is_duplicate = False
            for j in range(len(unique_masks)):
                iou = self.calculate_mask_iou(masks[i], unique_masks[j])
                if iou > 0.95:  
                    is_duplicate = True
                    break

            if not is_duplicate:
                xmin, ymin, xmax, ymax = boxes[i]
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                bbox_area = bbox_width * bbox_height
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else float('inf')

                if aspect_ratio > max_aspect_ratio or aspect_ratio < 1 / max_aspect_ratio:
                    continue
                if bbox_area < min_bbox_area:
                    continue
                if masks[i].sum().item() < min_mask_area:
                    continue

                unique_boxes.append(boxes[i])
                unique_masks.append(masks[i])
                unique_labels.append(labels[i])

        if len(unique_boxes) == 0:
            return target

        target['boxes'] = torch.stack(unique_boxes)
        target['masks'] = torch.stack(unique_masks)
        target['labels'] = torch.tensor(unique_labels, dtype=torch.int64)

        return target

    def calculate_mask_iou(self, mask1, mask2):
        intersection = torch.sum((mask1 & mask2).float())
        union = torch.sum((mask1 | mask2).float())
        iou = intersection / union

        return iou