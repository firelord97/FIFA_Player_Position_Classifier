# ⚽ FIFA Player Position Classification

This repository contains an analysis of supervised learning algorithms for classifying player positions in the EA Sports FC 24 dataset. We compare the performance of Neural Networks, Support Vector Machines (SVM), and k-Nearest Neighbors (k-NN) on this multiclass classification problem, revealing insights into the capabilities and limitations of each model when dealing with an imbalanced dataset of player statistics.

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#-dataset-overview)
3. [Models and Algorithms](#-models-and-algorithms)
4. [Results](#-results)
5. [Installation and Setup](#-installation-and-setup)
6. [How to Run](#-how-to-run)
7. [Future Work](#-future-work)
8. [References](#-references)

---

## Introduction

This project aims to classify soccer players' positions based on their in-game statistics using three supervised learning algorithms: Neural Networks, Support Vector Machines (SVM), and k-Nearest Neighbors (k-NN). The classification task involves predicting the primary position of a player, such as goalkeeper, midfielder, or forward, based on attributes like physical stats, skills, and playing style.

---

## Dataset Overview

The dataset, obtained from Kaggle, includes player statistics from the EA Sports FC 24 video game. The dataset features 46 attributes for each player and covers 15 distinct positions. The positions are categorized into three groups: Defense, Midfield, and Attack, with specific roles like Goalkeeper (GK), Central Midfielder (CM), and Striker (ST).

![Player Positions](fifa_positions.png)  
*Figure 1: General field positions of soccer players.*

- **Positions with Higher Counts**: Central Back (CB), Striker (ST), and Central Midfielder (CM) are well-represented.
- **Positions with Lower Counts**: Central Forward (CF), Left Wing-Back (LWB), and Right Wing-Back (RWB) are less frequent.

---

## Models and Algorithms

The project employs three algorithms for the classification task:

1. **Neural Networks (NN)**: 
   - Architecture: Two hidden layers with 64 nodes each, using activation functions like ReLU and Tanh.
   - Optimizer: Adam, with different learning rates.
   - Cross-validation: 5-fold cross-validation with early stopping.

2. **Support Vector Machines (SVM)**:
   - Kernels: Linear and Radial Basis Function (RBF).
   - Regularization parameter (C): Values of 0.1, 1, and 10.

3. **k-Nearest Neighbors (k-NN)**:
   - Hyperparameter tuning: Number of neighbors (k) set to 5 and 10.
   - Distance metric: Euclidean.

---

## Results

The analysis showed varying success across models:
- **Neural Networks**: Achieved high accuracy on distinct positions like GK and CB but struggled with overlapping roles such as LWB and LB.
- **Support Vector Machines**: Outperformed NNs in some cases, especially for linear separable data, but showed overfitting tendencies.
- **k-Nearest Neighbors**: Lagged behind other models, with a tendency to misclassify similar positions due to the high dimensionality of the data.

![Player Position Count](player_position_count.png)  
*Figure 2: Number of players per position in the dataset.*

Refer to the detailed confusion matrices and accuracy graphs for more insights.

## Confusion Matrix

The confusion matrix below shows the classification performance of the best model, highlighting the accuracy for each position. The matrix illustrates which positions were correctly classified and where misclassifications occurred.

![Confusion Matrix](graphs/confusion_matrix_best_model_percentage.png)  
*Figure 3: Confusion matrix showing the classification accuracy for each player position.*

---



---

## Installation and Setup

To get started, clone the repository and set up the required dependencies.

### Prerequisites

Make sure you have the following installed:
- **Python 3.6+**
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow` (for NNs), `seaborn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/fifa-position-classification.git

## How to Run

1. **Data Preprocessing**: Run the `FIFA_preprocessing.ipynb` notebook to preprocess the raw dataset and generate the cleaned dataset.

2. **Model Training**: Choose a model to train:
   - **Neural Network**: Run the `FIFA_nn.ipynb` notebook.
   - **Support Vector Machine**: Run the `FIFA_svm.ipynb` notebook.
   - **k-Nearest Neighbors**: Run the `FIFA_knn.ipynb` notebook.

3. **Evaluation**: After training, the notebooks will automatically generate evaluation metrics and confusion matrices for analysis.

4. **Visualization**: The results, including accuracy graphs and confusion matrices, can be found in the `results/` folder.

---

## Future Work

- **Class Balancing**: Apply techniques like SMOTE or class weighting to address the imbalance.
- **Position Grouping**: Merge similar positions (e.g., LWB and LB) to simplify classification.
- **Ensemble Methods**: Implement ensemble learning techniques to boost performance.

---

## References

1. EA Sports FC 24 Player Dataset - [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
2. Guide to Soccer Positions and Formations - [GiveMeSport](https://www.givemesport.com/guide-to-football-soccer-positions-formations)
3. scikit-learn: Machine Learning in Python - [scikit-learn](https://scikit-learn.org/)

---

