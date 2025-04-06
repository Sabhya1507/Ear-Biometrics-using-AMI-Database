# Ear-Biometrics-using-AMI-Database
Evaluation of three different approaches (SVM, CNN, CNN ResNet) for classification on AMI Ear Database

Overview
This project presents a comparative study between traditional machine learning and deep learning techniques for ear biometric recognition. Using the AMI Ear Database, we implemented and evaluated three distinct approaches:

Support Vector Machine (SVM) with handcrafted features: Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and Scale-Invariant Feature Transform (SIFT).

A custom Convolutional Neural Network (CNN) trained on grayscale images.

A ResNet-based CNN trained on RGB images.

The study aims to examine performance in terms of accuracy, computational efficiency, and the impact of color information in image recognition.

Dataset
The AMI Ear Database consists of 700 high-resolution RGB images from 100 individuals, with 7 images per person, captured in controlled conditions.

Dataset URL:
https://webctim.ulpgc.es/research_works/ami_ear_database/

Methodology
1. Data Preprocessing
Resizing images to a fixed dimension.

Grayscale conversion for one of the CNN models.

Normalization of pixel values (scaled between 0 and 1).

Data augmentation (rotation, flipping, and brightness adjustment).

2. Model Implementations
Approach 1: SVM with Handcrafted Feature Extraction
Feature Descriptors: HOG, LBP, and SIFT.

Features normalized and concatenated into a single feature vector.

Classifier: Linear Support Vector Machine (SVM).

Evaluation: 5-fold cross-validation.

Accuracy: ~60%

Advantages:

Lightweight and interpretable.

Performs reasonably on small datasets.

Disadvantages:

Struggles with lighting and pose variations.

Manual feature engineering required.

Approach 2: Custom CNN on Grayscale Images
Architecture: 3 convolutional layers, Batch Normalization, MaxPooling, followed by fully connected layers.

Optimizer: Adam with learning rate scheduling.

Training on grayscale images only.

Accuracy: ~79%

Advantages:

Learns feature representations automatically.

Requires less memory compared to RGB.

Disadvantages:

Accuracy limited by grayscale information.

Less robust to variations compared to deeper models.

Approach 3: ResNet-based CNN on RGB Images
Architecture: Pretrained ResNet model fine-tuned on RGB ear images.

Training with image augmentation techniques.

Accuracy: ~95.5%

Advantages:

High accuracy and generalizability.

Robust to variations in image conditions.

Disadvantages:

Higher computational requirements.

Needs larger datasets or transfer learning to perform well.

Results
Model	Feature Type	Accuracy
SVM (HOG+LBP+SIFT)	Handcrafted	60.0%
Custom CNN	Grayscale Images	79.0%
ResNet CNN	RGB Images	95.5%
Contributions
Performed end-to-end implementation of traditional and deep learning models.

Evaluated models using standard performance metrics (accuracy, precision, recall).

Analyzed the effect of grayscale vs RGB on CNN performance.

Explored practical trade-offs between performance and complexity for deployment scenarios.

License
This project is for academic use and subject to dataset licensing terms. Refer to the AMI Ear Database license for more details.
