# IEEE Bigdata Cup 2024: Building Extraction

## Team: Big Ban

This project is part of the IEEE Bigdata Cup 2024 competition, specifically focusing on building extraction using the Mask R-CNN model. Our team, Big Ban, is from Feng Chia University, Taiwan.

## Project Overview

The goal of this project is to develop a deep learning model for building extraction from satellite imagery. We use the Mask R-CNN model for object detection and segmentation tasks, specifically tailored for extracting building footprints.

## Data Preparation

The dataset is formatted according to the COCO (Common Objects in Context) dataset structure. You will need to prepare the dataset with the following folder structure:


```
coco/
├── annotations/
│   ├── train.json          # Annotation file for training data
│   └── val.json            # Annotation file for validation data
├── train/
│   └── image/              # Directory containing training images
│       └── (train_data)    # Replace (train_data) with your actual training images
└── val/
    └── image/              # Directory containing validation images
        └── (val_data)      # Replace (val_data) with your actual validation images
```


- **annotations/train.json**: Contains annotations for training data.
- **annotations/val.json**: Contains annotations for validation data.
- **train/image/**: Contains the training images.
- **val/image/**: Contains the validation images.

## Training

To train the model, use the `train.ipynb` notebook. You can adjust the parameters as needed within the notebook. Follow these steps:

1. Open `train.ipynb`.
2. Run each cell sequentially from top to bottom.

## Inference

For inference and generating submission files in CSV format, use the following structure for your test data:

```
test/
└── image/
    └── (inference_image)   # Replace (inference_image) with your actual test images
```


To perform inference, use the `pred.ipynb` notebook. Follow these steps:

1. Open `pred.ipynb`.
2. Run each cell sequentially from top to bottom.

The final output will be a `submission.csv` file ready for submission.

## Submission

The generated `submission.csv` file should be submitted as per the competition guidelines.

---
