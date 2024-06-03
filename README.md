# Titanic-Survival-Prediction

This repository contains a complete machine learning project aimed at predicting the survival of passengers on the Titanic. 
The project includes data preprocessing, model training, hyperparameter tuning, and evaluation using various machine learning techniques.

#Project Objective:
The objective of this project is to provide hands-on experience with a complete machine learning pipeline. You will read, preprocess, and analyze the Titanic dataset, experiment with different models and hyperparameters, and evaluate model performance. Advanced techniques such as cross-validation and hyperparameter tuning using GridSearchCV and RandomizedSearchCV are also implemented.

#Dataset:
The dataset used in this project is the Titanic dataset from Kaggle. It contains information about the passengers on the Titanic, including their demographics, class, and whether they survived the disaster.

#Steps Included
Data Loading:

The dataset is loaded from Kaggle using the provided URL.
Data Preprocessing:

Handling Missing Values: Missing values in numerical features are imputed using the median, and categorical features are imputed using the most frequent value.
Feature Scaling: Numerical features are scaled using standard scaling.
Categorical Encoding: Categorical features are converted into numerical format using one-hot encoding.
Duplicate Removal: Duplicate entries in the dataset are removed.
Model Training:

Two different models are experimented with: RandomForestClassifier and LogisticRegression.
Hyperparameter tuning is performed using GridSearchCV for RandomForestClassifier and RandomizedSearchCV for LogisticRegression.
Model Evaluation:

Cross-validation is implemented to ensure the robustness of model evaluation.
Models are evaluated based on precision and recall metrics.
The best model is selected based on cross-validation scores and performance on the test set.
Results:

Precision and recall scores for both models are calculated and compared.
Cross-validation scores are used to evaluate the stability and performance of the models across different subsets of the data.


#How to Run
1. Clone the Repository:
git clone https://github.com/Taiser-Riad/titanic-survival-prediction.git
cd titanic-survival-prediction

 
2. Install Dependencies:
Ensure you have Python installed along with the required libraries:
pip install pandas scikit-learn numpy

3. Run the Script:
Execute the script to perform data preprocessing, model training, and evaluation:

python titanic_survival_prediction.py


Key Files:
titanic_survival_prediction.py:
Main script containing the entire machine learning pipeline.
README.md:
Project description and instructions.
Results and Insights


RandomForestClassifier:

Cross-Validation Precision Scores: (Scores from cross-validation)
Mean Cross-Validation Precision: (Mean score)
Precision on Test Set: (Test set precision)
Recall on Test Set: (Test set recall)


LogisticRegression:

Cross-Validation Recall Scores: (Scores from cross-validation)
Mean Cross-Validation Recall: (Mean score)
Precision on Test Set: (Test set precision)
Recall on Test Set: (Test set recall)


Conclusion
This project demonstrates the end-to-end process of building a machine learning model, from data preprocessing to model evaluation and hyperparameter tuning.
The use of cross-validation ensures robust evaluation, and hyperparameter tuning helps in finding the best model configuration.

Feel free to explore, modify, and improve the code to suit your needs. Contributions and feedback are welcome!


Contact
For any questions or suggestions, feel free to contact [Taiser Riad] at [Taiser.Riad@hotmail.com].
