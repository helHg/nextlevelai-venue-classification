# Venue Image Classification Project
A Machine Learning and Deep Learning Project by NextLevelAI for COMP 6721 Applied Artificial Intelligence, Concordia University (Summer 2024).

This repository contains the code and resources for the Image Classification Project. The project involves classifying images into five different venue classes using both Decision Trees and Convolutional Neural Networks (CNNs). The dataset used for this project is a subset of the Places365-Standard dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebooks](#notebooks)
- [Acknowledgements](#acknowledgements)

## Project Overview

The main objective of this project is to classify images into one of five classes: `airplane_cabin`, `hockey_arena`, `movie_theater`, `staircase`, and `supermarket`. The project explores different machine learning and deep learning techniques, including:
- Supervised learning with Decision Trees
- Semi-supervised learning with Decision Trees
- Supervised learning with Convolutional Neural Networks (CNNs)

## Repository Structure

- `notebooks/`: Contains all the Jupyter notebooks used for this project.
  - `cnn.ipynb`: Contains the implementation of the CNN model.
  - `decision_tree_models.ipynb`: Contains the implementation of the supervised and semi-supervised Decision Tree models.
  - `image_preprocessing.ipynb`: Contains the code for splitting the `train_val` folder and augmenting the training set.
  - `project_raw_dataset_creation.ipynb`: Extracts the necessary images from the original Places365-Standard dataset and organizes them into `test` and `train_val` folders.
- `requirements.txt`: Lists all the Python libraries required to run the notebooks.

## Requirements

To run the notebooks, you need to have Python installed on your system. The required libraries are listed in the `requirements.txt` file.

## Installation

To install the required libraries, use the following command:
```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies specified in the `requirements.txt` file.

## Notebooks

### cnn.ipynb
This notebook contains the implementation of the CNN model used for image classification. The sections in this notebook are very descriptive and explicitly named to guide you through the CNN model's architecture, training, and evaluation process.

### decision_tree_models.ipynb
This notebook includes the implementation of both the supervised and semi-supervised Decision Tree models. Each section is well-documented and clearly describes the steps involved in training and evaluating the Decision Tree models.

### project_raw_dataset_creation.ipynb
This notebook extracts the images required for the project from the original Places365-Standard dataset, which can be downloaded from [this link](http://places2.csail.mit.edu/download-private.html). The extracted images are then organized into two folders: `test` and `train_val`.

### image_preprocessing.ipynb
This notebook handles the preprocessing of images, including splitting the `train_val` folder into training and validation sets and augmenting the training set. The sections in this notebook are highly descriptive, making the preprocessing steps easy to follow.

## Acknowledgements

This project uses the Places365-Standard dataset, which is provided by the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL). We thank the authors for making this dataset available for research purposes.

We also extend our gratitude to Rose and Professor Azarfar for their invaluable insights and support throughout this project.
