from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def fit(texts, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1_000, C=10).fit(X, y)
    return vec, clf

def score(vec, clf, texts):
    return clf.predict_proba(vec.transform(texts))[:, 1]