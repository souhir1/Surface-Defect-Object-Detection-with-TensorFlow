# Surface-Defect-Object-Detection-with-TensorFlow
This repository hosts a comprehensive project aimed at solving surface defect detection challenges using the NEU Surface Defect Database from Kaggle. The project is structured into two main tasks, each elaborately documented in separate Jupyter notebooks to provide a thorough understanding and guide through the process.

## Project Structure

- **Task 1 Notebook (`Task1_Data_Preparation_Object_Detection.ipynb`):** Focuses on dataset preparation, converting raw data into TFRecord format for optimal processing with MobileNetV2. This notebook covers data cleaning, augmentation, and transformation steps.
- **Task 3 Notebook (`task3-object-detection-mobilenetv2.ipynb`):** Delves into adapting and fine-tuning the MobileNetV2 model for surface defect detection. It discusses model architecture adjustments, optimization strategies, and the inference process tailored for object detection using TensorFlow.

## Detailed Documentation

### Task 1: Data Preparation for Object Detection

#### Overview

The initial stage of this project centers on preparing the dataset for object detection tasks, which is pivotal for structuring the data for efficient model training and evaluation.

#### Process

1. **Dataset Conversion:** Transformation of the dataset into the `tf.data.TFRecordDataset` format, employing a custom feature schema within `tf.train.Example` to accurately encapsulate data characteristics essential for object detection.
2. **Directory Structure Creation:** Organization of the dataset into two primary directories for streamlined access during training and testing:
   - **Train Directory:** Houses individual sample TFRecord files for the training phase.
   - **Test Directory:** Contains individual sample TFRecord files for testing and model evaluation.
3. **Data Adaptation for Model Input:** Adjustment of the dataset to meet the model's input specifications through preprocessing steps like image resizing, pixel value normalization, and label encoding. This ensures compatibility with the model’s input requirements, crucial for effective training and inference.

### Task 3: Object Detection with MobileNetV2

#### Overview

This task is focused on developing an object detection pipeline using a pre-trained MobileNetV2 backbone for feature extraction, supplemented with a custom neural network layer for the detection specifics.


1. **Model Choice and Architecture:** This selection was made after comparing several models, including Faster R-CNN and EfficientDet. MobileNetV2 was chosen for its seamless TensorFlow integration and its unique ability to perform both classification and bounding box regression in one shot, streamlining the object detection process. The architecture includes global average pooling and dropout layers to mitigate overfitting, with branches for bounding box regression and class prediction.
2. **Model Optimization:** Fine-tuning of MobileNetV2 specifically for the object detection task to ensure precise bounding box predictions and class identification. The model employs the Adam optimizer and uses separate loss functions for the bounding box and class outputs, incorporating strategies like early stopping and learning rate reduction to enhance training efficiency and model performance.

#### Training Outcomes

- The model underwent training for 100 epochs, with adaptive learning rate adjustments and early stopping mechanisms to prevent overfitting, concluding the training at epoch 57 to preserve the optimal weights from epoch 47. This strategy led to high accuracy in class predictions and minimal error in bounding box predictions, as highlighted in the performance metrics table below.

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Bounding Box Output Mean Squared Error | 0.0369 | 0.0344 | 0.0329 |
| Class Output Accuracy (%) | 99.49 | 99.31 | 99.46 |
| Loss | 0.0603 | 0.0522 | 0.0619 |


The training strategy and model optimization resulted in high accuracy and low error, demonstrating the effectiveness of the approach.

#### Future Work

-Given more time, I plan to integrate an SSD (Single Shot Multibox Detector) with the current MobileNetV2 model to further enhance the precision of bounding box predictions. This combination is expected to leverage the speed and efficiency of MobileNetV2 with the accurate localization capabilities of SSD, potentially offering significant improvements in detecting and precisely mapping objects within images

