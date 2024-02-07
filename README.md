# FireWatch: Satellite-Based Forest Fire Detection System

## Overview

FireWatch is a state-of-the-art system that uses machine learning and image processing to detect forest fires from satellite images. Created as part of a machine learning course at Hanyang University, this project aims to contribute to early fire detection efforts, potentially saving vast areas of forests and preventing the loss of wildlife and property. The system analyzes satellite imagery to distinguish fire incidents, facilitating timely responses from authorities and environmental agencies.

## Features

- **Advanced Image Processing**: Utilizes OpenCV and skimage for image preprocessing and feature extraction.
- **Robust Machine Learning Models**: Incorporates various models, including RandomForestClassifier, GradientBoostingClassifier, and MLPClassifier, to ensure accurate fire detection.
- **Deep Learning Integration**: Leverages TensorFlow and Keras for enhanced image classification through Convolutional Neural Networks (CNNs).
- **Comprehensive Dataset**: Includes a curated dataset with labeled satellite images of fire and non-fire scenarios to train and evaluate the models.

## Installation

To set up the FireWatch system, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python installed, along with the necessary libraries (numpy, matplotlib, OpenCV, scikit-learn, TensorFlow, Keras). You can navigate to the project directory and install the requirements using `pip install -r requirements.txt` if needed.

## Usage

To run the fire detection system, execute the main script (`python Code.py`). The script will process the satellite images in the dataset and output the classification results, including accuracy metrics and confusion matrices.

## Contributing

Contributions to FireWatch are welcome! If you're interested in improving the detection algorithms or adding new features, please feel free to fork the repository and submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the instructors and peers at Hanyang University for their support and guidance throughout the development of this project.
