from sklearn.feature_extraction.text import CountVectorizer

def getCountVectors(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized