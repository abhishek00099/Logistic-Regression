Breast Cancer Diagnosis using Logistic Regression
This project demonstrates the implementation of a binary classification model using Logistic Regression to predict whether a breast tumor is malignant (M) or benign (B). The model is built using Python with popular data science libraries like Scikit-learn, Pandas, and Matplotlib.

Project Overview
The primary goal of this project is to build an accurate and interpretable machine learning model for breast cancer classification. The project covers the complete machine learning workflow:

Data Loading and Preprocessing: Loading the dataset and preparing it for modeling.

Model Training: Implementing and training a Logistic Regression classifier.

Model Evaluation: Assessing the model's performance using standard classification metrics.

Threshold Tuning: Analyzing the trade-off between precision and recall to optimize the classification threshold.

Dataset
The model is trained on the Wisconsin Breast Cancer dataset. This dataset contains 569 instances and 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Features: Characteristics of the cell nuclei (e.g., radius_mean, texture_mean, smoothness_mean).

Target Variable: diagnosis, which is binary (M = Malignant, B = Benign).

Methodology
1. Data Preparation
The diagnosis column is converted from categorical ('M'/'B') to numerical (1/0). The dataset is then split into training (80%) and testing (20%) sets.

2. Feature Scaling
Features are standardized using StandardScaler from Scikit-learn. This ensures that all features have a mean of 0 and a standard deviation of 1, which helps the logistic regression algorithm to converge faster and perform better.

3. Logistic Regression and the Sigmoid Function
A Logistic Regression model is trained on the scaled training data. This model uses the sigmoid function to map the linear combination of input features to a probability value between 0 and 1. This probability is then used to classify the tumor.




<img width="2048" height="2047" alt="image" src="https://github.com/user-attachments/assets/e6023181-0154-4e51-bd3e-79588dc3ffda" />






The Sigmoid Function, which maps any real value into a value between 0 and 1.

4. Evaluation
The model's performance is evaluated on the test set using the following metrics:

Confusion Matrix: Provides a summary of true positives, true negatives, false positives, and false negatives.

Precision & Recall: Measures the accuracy of positive predictions and the ability to identify all actual positives, respectively.

ROC-AUC Score: The Area Under the Receiver Operating Characteristic Curve measures the model's ability to distinguish between classes. An AUC score of 1.0 indicates a perfect classifier.

ROC Curve illustrating the trade-off between the True Positive Rate and False Positive Rate.

5. Threshold Tuning
By default, logistic regression uses a probability threshold of 0.5 for classification. However, for medical applications, it's often crucial to minimize false negatives. We can adjust this threshold to find an optimal balance between precision and recall for our specific problem.

The relationship between the classification threshold and Precision/Recall scores.

How to Run
Prerequisites: Ensure you have Python and the following libraries installed:

pip install scikit-learn pandas matplotlib

Dataset: Place the data.csv file in the same directory as the script.

Execute: Run the Python script from your terminal:

python classification_script.py
