import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from collections import Counter

import KeyThemes as kt

import sys
sys.path.append('./Objects')
import Constants as cn
import Variables as vr

with open(r".\TopicModelling\TFIDFVectorizers\TFIDFVectorizer1_LDA.pkl", "rb") as file:
    vectorizer1 = pickle.load(file)

with open(r".\TopicModelling\LDAModels\LDA_Model1.pkl", "rb") as file:
    LDAModel1 = pickle.load(file)

with open(r".\TopicModelling\TFIDFVectorizers\TFIDFVectorizer2_LDA.pkl", "rb") as file:
    vectorizer2 = pickle.load(file)

with open(r".\TopicModelling\LDAModels\LDA_Model2.pkl", "rb") as file:
    LDAModel2 = pickle.load(file)

with open(r".\TopicModelling\TFIDFVectorizers\TFIDFVectorizer3_LDA.pkl", "rb") as file:
    vectorizer3 = pickle.load(file)

with open(r".\TopicModelling\LDAModels\LDA_Model3.pkl", "rb") as file:
    LDAModel3 = pickle.load(file)

with open(r".\TopicModelling\TFIDFVectorizers\TFIDFVectorizer4_LDA.pkl", "rb") as file:
    vectorizer4 = pickle.load(file)

with open(r".\TopicModelling\LDAModels\LDA_Model4.pkl", "rb") as file:
    LDAModel4 = pickle.load(file)

with open(r".\TopicModelling\TFIDFVectorizers\TFIDFVectorizer5_LDA.pkl", "rb") as file:
    vectorizer5 = pickle.load(file)

with open(r".\TopicModelling\LDAModels\LDA_Model5.pkl", "rb") as file:
    LDAModel5 = pickle.load(file)


def getKeyThemes():
    df = vr.dataFrameProcessed
    
    keyTheme_HN = ''
    keyTheme_N = ''
    keyTheme_Neu = ''
    keyTheme_P = ''
    keyTheme_HP = ''

    reviews_HN = df[df['Sentiment'] == cn.SENTIMENT1]
    if reviews_HN.empty:
        keyTheme_HN = kt.NO_THEME
    else:
        tokenized_Reviews_HN = reviews_HN['TokenizedReviews'].tolist()
        vectors_HN = vectorizer1.transform(tokenized_Reviews_HN)
        themes_Distribution_HN = LDAModel1.transform(vectors_HN)
        themes_Frequent_HN = np.argmax(themes_Distribution_HN, axis=1)
        theme_Main_HN = Counter(themes_Frequent_HN).most_common(1)[0][0]
        match theme_Main_HN:
            case 0:
                keyTheme_HN = kt.HN_THEME_0
            case 1:
                keyTheme_HN = kt.HN_THEME_1
            case 2:
                keyTheme_HN = kt.HN_THEME_2
    
    reviews_N = df[df['Sentiment'] == cn.SENTIMENT2]
    if reviews_N.empty:
        keyTheme_N = kt.NO_THEME
    else:
        tokenized_Reviews_N = reviews_N['TokenizedReviews'].tolist()
        vectors_N = vectorizer2.transform(tokenized_Reviews_N)
        themes_Distribution_N = LDAModel2.transform(vectors_N)
        themes_Frequent_N = np.argmax(themes_Distribution_N, axis=1)
        theme_Main_N = Counter(themes_Frequent_N).most_common(1)[0][0]
        match theme_Main_N:
            case 0:
                keyTheme_N = kt.N_THEME_0
            case 1:
                keyTheme_N = kt.N_THEME_1
            case 2:
                keyTheme_N = kt.N_THEME_2
    
    reviews_Neu = df[df['Sentiment'] == cn.SENTIMENT3]
    if reviews_Neu.empty:
        keyTheme_Neu = kt.NO_THEME
    else:
        tokenized_Reviews_Neu = reviews_Neu['TokenizedReviews'].tolist()
        vectors_Neu = vectorizer3.transform(tokenized_Reviews_Neu)
        themes_Distribution_Neu = LDAModel3.transform(vectors_Neu)
        themes_Frequent_Neu = np.argmax(themes_Distribution_Neu, axis=1)
        theme_Main_Neu = Counter(themes_Frequent_Neu).most_common(1)[0][0]
        match theme_Main_Neu:
            case 0:
                keyTheme_Neu = kt.NEU_THEME_0
            case 1:
                keyTheme_Neu = kt.NEU_THEME_1
            case 2:
                keyTheme_Neu = kt.NEU_THEME_2

    reviews_P = df[df['Sentiment'] == cn.SENTIMENT4]
    if reviews_P.empty:
        keyTheme_P = kt.NO_THEME
    else:
        tokenized_Reviews_P = reviews_P['TokenizedReviews'].tolist()
        vectors_P = vectorizer4.transform(tokenized_Reviews_P)
        themes_Distribution_P = LDAModel4.transform(vectors_P)
        themes_Frequent_P = np.argmax(themes_Distribution_P, axis=1)
        theme_Main_P = Counter(themes_Frequent_P).most_common(1)[0][0]
        match theme_Main_P:
            case 0:
                keyTheme_P = kt.P_THEME_0
            case 1:
                keyTheme_P = kt.P_THEME_1
            case 2:
                keyTheme_P = kt.P_THEME_2
    
    reviews_HP = df[df['Sentiment'] == cn.SENTIMENT5]
    if reviews_HP.empty:
        keyTheme_HP = kt.NO_THEME
    else:
        tokenized_Reviews_HP = reviews_HP['TokenizedReviews'].tolist()
        vectors_HP = vectorizer5.transform(tokenized_Reviews_HP)
        themes_Distribution_HP = LDAModel5.transform(vectors_HP)
        themes_Frequent_HP = np.argmax(themes_Distribution_HP, axis=1)
        theme_Main_HP = Counter(themes_Frequent_HP).most_common(1)[0][0]
        match theme_Main_HP:
            case 0:
                keyTheme_HP = kt.HP_THEME_0
            case 1:
                keyTheme_HP = kt.HP_THEME_1
            case 2:
                keyTheme_HP = kt.HP_THEME_2

    return keyTheme_HN, keyTheme_N, keyTheme_Neu, keyTheme_P, keyTheme_HP