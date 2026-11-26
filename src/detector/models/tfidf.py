from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def fit(texts, y, C=10, ngram_range=(1, 2), min_df=3, sublinear_tf=True):
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1_000, C=C).fit(X, y)
    return vec, clf

def score(vec, clf, texts):
    return clf.predict_proba(vec.transform(texts))[:, 1]