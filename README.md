# SMS Spam Detection

## Overview
This project builds a machine learning model to classify SMS messages as spam or ham (not spam). It uses Natural Language Processing techniques to convert text into numerical features and applies a Multinomial Naive Bayes classifier for prediction. Two different vectorization approaches, CountVectorizer and TF-IDF, are implemented and compared.

## Dataset
The dataset used is spam.csv, which contains 5572 SMS messages labeled as either ham or spam. The dataset has 4825 ham messages and 747 spam messages.

## Project Workflow

### 1. Data Loading
The dataset is loaded using pandas with latin-1 encoding. Only the relevant columns are kept and renamed to label and text for clarity.

### 2. Text Preprocessing
The text data is cleaned by converting all characters to lowercase, removing special characters and punctuation, and stripping extra whitespace to prepare it for vectorization.

### 3. Label Encoding
The label column is converted into numerical form where ham is mapped to 0 and spam is mapped to 1.

### 4. Train Test Split
The dataset is split into training and testing sets with an 80 to 20 ratio to evaluate model performance on unseen data.

### 5. Feature Extraction
Two vectorization techniques are used to convert text into numerical features
CountVectorizer which represents text based on word frequency
TF-IDF Vectorizer which represents text based on word importance across the dataset
Both are limited to the top 2000 features with English stop words removed.

### 6. Model Training
A Multinomial Naive Bayes model is trained separately on the CountVectorizer features and the TF-IDF features to compare their performance.

### 7. Model Evaluation
Both models are evaluated using accuracy score, confusion matrix, and classification report to measure precision, recall, and f1-score for each class.

### 8. Feature Importance Visualization
The top 10 most influential words associated with spam messages are visualized using a horizontal bar chart based on their learned probability contribution.

## Results

CountVectorizer Model
Accuracy 97.84 percent

TF-IDF Model
Accuracy 97.39 percent

The CountVectorizer based model performed slightly better than the TF-IDF based model on this dataset.

## Technologies Used
Python
Pandas
NumPy
Scikit learn
Matplotlib

## How to Run
1. Clone this repository
2. Install the required libraries using pip install numpy pandas matplotlib scikit learn
3. Place the spam.csv file in the same directory as the notebook
4. Open the notebook file SMS_Spam_Detection.ipynb using Jupyter Notebook or any compatible environment
5. Run all cells sequentially to reproduce the results

## Future Improvements
Try other machine learning models such as Support Vector Machines or Logistic Regression
Perform hyperparameter tuning to improve accuracy
Add cross validation for more reliable performance evaluation
Handle class imbalance since spam messages are fewer than ham messages
Experiment with n-grams and additional text preprocessing steps such as stemming or lemmatization

## Author
This project was developed as part of a machine learning learning exercise focused on natural language processing and text classification.
