from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import torch.nn.functional as NNF
class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor
    ) -> Tensor:
        image = F.pil_to_tensor(image)

        return image
    
def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-2)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints_vertically(keypoints, height)
                    target["keypoints"] = keypoints
        return image, target

def _flip_coco_person_keypoints_vertically(keypoints: Tensor, height: int) -> Tensor:
    flipped_parts = (0, 1, 4, 5, 2, 3, 6, 7, 8, 11, 12, 9, 10, 13, 14, 15, 16)
    flipped_keypoints = keypoints.clone()
    
    flipped_keypoints[..., 1] = height - flipped_keypoints[..., 1]
    
    for i, j in enumerate(flipped_parts):
        flipped_keypoints[..., i, :] = keypoints[..., j, :]
    
    return flipped_keypoints

class RandomCropWithBBoxes(nn.Module):
    def __init__(self, min_crop_size=200, max_crop_size=400, min_mask_area=600, min_bbox_area=600, max_aspect_ratio=3.5, p=0.25):
        super(RandomCropWithBBoxes, self).__init__()
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.min_mask_area = min_mask_area
        self.min_bbox_area = min_bbox_area
        self.max_aspect_ratio = max_aspect_ratio
        self.p = p

    def forward(self, image, target):
        if torch.rand(1).item() > self.p:
            return image, target
        
        width, height = image.size
        assert width == height, "Image must be square."

        new_size = 256 #torch.randint(self.min_crop_size, self.max_crop_size, (1,)).item()

        if width <= new_size or height <= new_size:
            return image, target

        left = torch.randint(0, width - new_size, (1,)).item()
        top = torch.randint(0, height - new_size, (1,)).item()

        image = F.crop(image, top, left, new_size, new_size)

        boxes = target['boxes']
        masks = target['masks']

        new_boxes = []
        new_masks = []
        new_labels = []

        for i, (box, mask) in enumerate(zip(boxes, masks)):
            xmin, ymin, xmax, ymax = box
            new_xmin = max(0, xmin - left)
            new_ymin = max(0, ymin - top)
            new_xmax = min(new_size, xmax - left)
            new_ymax = min(new_size, ymax - top)

            if new_xmin < new_xmax and new_ymin < new_ymax:
                bbox_width = new_xmax - new_xmin
                bbox_height = new_ymax - new_ymin
                bbox_area = bbox_width * bbox_height
                origin_area_ratio = bbox_area / (xmax - xmin) * (ymax - ymin)
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else float('inf')
                if aspect_ratio > self.max_aspect_ratio or aspect_ratio < 1 / self.max_aspect_ratio:
                    # print('Aspect ratio is too large or too small.')
                    continue
                if bbox_area < self.min_bbox_area or origin_area_ratio < 0.3:
                    # print('Bbox area is too small.')
                    continue
                               
                cropped_mask = F.crop(mask, top, left, new_size, new_size)

                if cropped_mask.sum().item() >= self.min_mask_area:
                    new_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
                    new_masks.append(cropped_mask)
                    new_labels.append(target['labels'][i])
                else:
                    # print('Mask area is too small.')
                    continue

        target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
        try:
            target['masks'] = torch.stack(new_masks)
        except:
            target['masks'] = torch.tensor(new_masks)
        target['labels'] = torch.tensor(new_labels, dtype=torch.int64)

        return image, target

class MosaicWithBBoxes(nn.Module):
    def __init__(self, img_size=256, mosaic_border=(-256, -256)):
        super(MosaicWithBBoxes, self).__init__()
        self.img_size = img_size  
        self.mosaic_border = mosaic_border  
        self.keep_iou_threshold = 0.7  
        self.crop = RandomCropWithBBoxes(min_crop_size=256, max_crop_size=256, min_mask_area=100, min_bbox_area=100, max_aspect_ratio=3.5, p=1)

    def forward(self, imgs, all_boxes, all_labels, all_masks):

        yc, xc = 256, 256 #(int(random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border)

        img4 = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)

        new_boxes = []
        new_labels = []
        new_masks = []

        for i, (img, boxes, labels, masks) in enumerate(zip(imgs, all_boxes, all_labels, all_masks)):
            img, target = self.crop(img, {'boxes': boxes, 'labels': labels, 'masks': masks})
            boxes = target['boxes']
            labels = target['labels']
            masks = target['masks']
            w, h = img.size

            if i == 0:  
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img = np.array(img)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            boxes, labels, masks = self.adjust_boxes_and_masks(boxes, labels, masks, padw, padh, x1a, y1a, x2a, y2a, w, h)

            new_boxes.extend(boxes)
            new_labels.extend(labels)
            new_masks.extend([self.adjust_masks(mask, padw, padh, x1a, y1a, x2a, y2a, w, h) for mask in masks])

        new_boxes = np.clip(new_boxes, 0, 2 * self.img_size)

        try:
            target = {
            'boxes': torch.tensor(new_boxes, dtype=torch.float32),
            'labels': torch.tensor(new_labels, dtype=torch.int64),
            'masks': torch.stack([mask.clone().detach().to(torch.uint8) for mask in new_masks])
            }
        except:
            target = {
            'boxes': torch.tensor(new_boxes, dtype=torch.float32),
            'labels': torch.tensor(new_labels, dtype=torch.int64),
            'masks': torch.tensor([mask.clone().detach().to(torch.uint8) for mask in new_masks])
            }

        return img4, target
    
    def adjust_boxes_and_masks(self, boxes, labels, masks, padw, padh, x1a, y1a, x2a, y2a, w, h):
        adjusted_boxes = []
        adjusted_labels = []
        adjusted_masks = []

        for box, label, mask in zip(boxes, labels, masks):
            xmin, ymin, xmax, ymax = box
            original_area = (xmax - xmin) * (ymax - ymin)  

            xmin = max(xmin + padw, 0)
            ymin = max(ymin + padh, 0)
            xmax = min(xmax + padw, 2 * self.img_size)
            ymax = min(ymax + padh, 2 * self.img_size)

            if xmin >= xmax or ymin >= ymax:  
                continue

            adjusted_area = (xmax - xmin) * (ymax - ymin)  
            area_ratio = adjusted_area / original_area

            if area_ratio < self.keep_iou_threshold or adjusted_area < 100:  
                continue

            adjusted_boxes.append([xmin, ymin, xmax, ymax])
            adjusted_labels.append(label)
            adjusted_masks.append(mask)  

        return adjusted_boxes, adjusted_labels, adjusted_masks



    def adjust_masks(self, mask, padw, padh, x1a, y1a, x2a, y2a, w, h):
        left_pad = padw
        right_pad = 2 * self.img_size - (x2a - x1a) - padw
        top_pad = padh
        bottom_pad = 2 * self.img_size - (y2a - y1a) - padh

        final_height = mask.shape[0] + top_pad + bottom_pad
        final_width = mask.shape[1] + left_pad + right_pad

        if final_height != 2 * self.img_size:
            bottom_pad = 2 * self.img_size - (mask.shape[0] + top_pad)
        if final_width != 2 * self.img_size:
            right_pad = 2 * self.img_size - (mask.shape[1] + left_pad)
        
        return NNF.pad(mask, (left_pad, right_pad, top_pad, bottom_pad))

class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 0.5,  
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials
        self.p = p  

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        
        if torch.rand(1).item() >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target

class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target
    
class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias=True,
        p: float = 0.5,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)
        
        if torch.rand(1) >= self.p:
            return image, target

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.antialias)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST,
                    antialias=self.antialias,
                )

        return image, target
    
class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target

class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target
