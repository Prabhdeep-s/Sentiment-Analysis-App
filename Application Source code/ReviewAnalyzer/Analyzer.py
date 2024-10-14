import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model._logistic import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from fasttext import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

import sys
sys.path.append('./Tokenizer')
sys.path.append('./Objects')
import CustomTokenizer as ctn
import Constants as cn
import Variables as vr

fastTextVectorizer = load_model(r".\Vectorizer\FastTextVectorizer.bin")

with open(r".\Classifier\LR_FastText_Classifier.pkl", "rb") as file:
    LRClassifier = pickle.load(file)

def analyze_file(file_path):
    # Get the file extension
    ext = os.path.splitext(file_path)[1]

    if ext == '.csv':
        dataFrame = pd.read_csv(file_path)
    elif ext == '.xlsx' or ext == '.xls':
        dataFrame = pd.read_excel(file_path)
    else:
        raise RuntimeError('File extension not recognized')

    dataFrame = dataFrame.dropna(how='any', axis=0) #remove null values
    reviews = dataFrame['Review'].values
    if len(reviews) == 0:
        return pd.DataFrame()
    tokenizedReviews = [ctn.preprocess_text(text) for text in reviews]

    vetorizedtexts = [fastTextVectorizer.get_sentence_vector(tokenList) for tokenList in tokenizedReviews]
    #y = dataFrame['Rating'].values

    predictedSentiments = LRClassifier.predict(vetorizedtexts)

    # Add Sentiment column in the dataframe
    dataFrame['Sentiment Score'] = predictedSentiments
    scoresDF = pd.DataFrame({'Scores': predictedSentiments})
    dataFrame['Sentiment'] = scoresDF['Scores'].apply(ConvertScoresToSentiments)

    dataFrameWithTokens = dataFrame.copy()
    dataFrameWithTokens['TokenizedReviews'] = tokenizedReviews
    vr.dataFrameProcessed = dataFrameWithTokens

    return dataFrame

def analyze_text(text):
    tokenizedText = ctn.preprocess_text(text)
    vetorizedtext = [fastTextVectorizer.get_sentence_vector(tokenizedText)]
    predictedSentiment = LRClassifier.predict(vetorizedtext)
    return ConvertScoresToSentiments(predictedSentiment[0])

def ConvertScoresToSentiments(score):
    match score:
        case 1:
            return cn.SENTIMENT1
        case 2:
            return cn.SENTIMENT2
        case 3:
            return cn.SENTIMENT3
        case 4:
            return cn.SENTIMENT4
        case 5:
            return cn.SENTIMENT5