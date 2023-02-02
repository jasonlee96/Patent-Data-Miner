import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import pickle
import joblib
from os.path import dirname, abspath, join


class PatentMining:
    def __init__(self):
        ROOT_DIR = dirname(dirname(abspath(__file__)))
        filename = join(ROOT_DIR, 'finalized_vectorizer.pkl')
        self.cv = joblib.load(filename)

        filename = join(ROOT_DIR, 'finalized_tfidf.pkl')
        self.vectorizer = joblib.load(filename)

        self.df = pd.read_csv("imported_data.csv")

        filename = join(ROOT_DIR, 'finalized_knn.pkl')
        self.knn = joblib.load(filename)

    def get_query(self, term):
        result = []
        count = self.cv.transform([term])

        test = self.vectorizer.transform(count)
        test_norm = normalize(test)
        test_array = test_norm.toarray()
        if np.all((test_array == 0)):
            print("Not found")
            return result
        queries = self.knn.kneighbors(test_array, return_distance=False)
        for ind in queries[0]:
            result.append([self.df.iloc[ind].Title, self.df.iloc[ind]["Patent Number"]])
        return result
