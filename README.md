# Vehicle Detection Project

## Overview

This project aims to develop a computer vision system for detecting vehicles in images and videos using machine learning techniques. The main goal is to train a classifier that can accurately distinguish between vehicles (cars, trucks, etc.) and non-vehicles (background objects, pedestrians, etc.).

## Dependencies

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- XGBoost
- Moviepy
- Pandas
- Seaborn

## Dataset

The project utilizes two datasets: one for vehicles and another for non-vehicles. The vehicle images are obtained from the `./dataset/vehicles/` directory, while the non-vehicle images are located in `./dataset/non-vehicles/`.

## Features

The following features are extracted from the images to train the classifier:

- **Spatial Binning**: Resizing the image to a smaller spatial resolution and using the raw pixel values as features.
- **Color Histograms**: Computing the color histogram of the image in different color spaces (e.g., RGB, HSV).
- **Histogram of Oriented Gradients (HOG)**: Computing the HOG features, which capture the local gradient information in the image.

## Model Training

The project trains a machine learning model using the extracted features. The code supports different models, such as Linear Support Vector Machines (SVMs), Random Forests, Decision Trees, and XGBoost. The training data is split into training and testing sets, and the model's performance is evaluated on the testing set.

## Vehicle Detection Pipeline

The project implements a pipeline for detecting vehicles in images and videos. This pipeline involves the following steps:

1. **Sliding Window**: A sliding window technique is used to extract regions of interest (ROIs) from the input image or video frame.
2. **Feature Extraction**: For each ROI, features are extracted using the same techniques employed during training.
3. **Classification**: The trained model classifies each ROI as either a vehicle or a non-vehicle.
4. **Heat Map and Thresholding**: A heat map is created by aggregating the positive detections across multiple sliding window positions. Thresholding is applied to the heat map to remove false positives.
5. **Bounding Box Generation**: Final bounding boxes are generated from the thresholded heat map, indicating the locations of detected vehicles.

## Usage

1. Ensure that all necessary dependencies are installed.
2. Prepare the dataset by placing vehicle and non-vehicle images in the respective directories (`./dataset/vehicles/` and `./dataset/non-vehicles/`).
3. Configure the parameters in the `Config` module according to your requirements.
4. Run the main script or execute the relevant code cells in the provided notebook.

The output will include visualizations of the detected vehicles on test images or a processed video file with bounding boxes around the detected vehicles.

## Visualization

The code provides visualizations to better understand the data and model performance, including:

- Feature distribution
- Feature correlation
- Feature importance (if supported by the trained model)
- Decision tree representation (if a Decision Tree model is used)


## Authors

- [MihirBibhuty](https://github.com/MihirBibhuty)
- [SahilVishwakarma](https://github.com/SVatghub)
- [RamKumar](https://github.com/rkrathore459954)
- [PrateekMishra](https://github.com/prateek-m20)
