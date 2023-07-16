##Stock Market Prediction with News Headlines
#Overview
This project aims to predict the rise or decrease of the DJIA (Dow Jones Industrial Average) Adj Close value based on news headlines. The dataset provided contains news headlines and stock market data, with the goal of building a machine learning model to predict the stock market movement based on the news.

#Dataset
The dataset consists of multiple columns, including the date, the label indicating whether the DJIA Adj Close value will rise or decrease, and news headlines from "Top1" to "Top25". The data has been preprocessed and cleaned for analysis.

#Problem Statement and Objective
The objective of this project is to build a machine learning model that can predict the rise or decrease of the DJIA Adj Close value based on the news headlines. It is a binary classification task, where the model will predict either a "1" for rise/stay the same or a "0" for decrease.

#Methodology and Approach
The project follows the following methodology:

Data Cleaning and Preprocessing: The dataset is cleaned and preprocessed to handle any missing values, outliers, or inconsistencies.

Text Processing and Feature Extraction: Natural Language Processing (NLP) techniques are applied to process the text data from the news headlines. Text preprocessing steps, such as tokenization, stemming, and removing stop words, are performed. Feature extraction methods, such as bag-of-words or TF-IDF, are used to represent the headlines as numerical features.

Model Selection and Training: Various machine learning algorithms, including Convolutional Neural Networks (CNN) and Graph Convolutional Networks (GCN), are trained on the processed data. The CNN model is trained on the vectorized data using the TensorFlow library, while the GCN model is trained using the PyTorch Geometric library.

Model Evaluation and Optimization: The trained models are evaluated using various metrics such as accuracy, R2 score, and precision. The models are optimized through hyperparameter tuning and experimentation to improve their performance.

#Results
After training and evaluation, the models achieved the following results:

CNN Model:

Accuracy: 0.49
GCN Model:

Accuracy: 0.972
R2 Score: 0.889
Precision: 0.975
#Conclusion and Future Work
In conclusion, the GCN model outperformed the CNN model in terms of accuracy, R2 score, and precision. The GCN model achieved a high accuracy of 0.972 and showed promising performance in predicting the rise or decrease of the DJIA Adj Close value based on the news headlines.

Future work for this project could involve exploring other advanced machine learning models and techniques, such as transformer-based models like BERT, to further improve the prediction accuracy. Additionally, incorporating sentiment analysis and market indicators as additional features may enhance the prediction capabilities.
