from sklearn.feature_extraction.text import TfidfVectorizer

def getTFIDFVectors(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized