# Import necessary modules
import xml.etree.ElementTree as ET
import re
import pandas as pd
from PatentMiner.preprocessing import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import joblib
import os

MAX_DF = 0.4
MIN_DF = 10
N_NEIGHBORS = 20
class KNNModel:
    def train(self):
        print("Train process started, may took more than an hours")
        xml_doc = []
        xml_string = ""
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(ROOT_DIR, "data")
        for root, dirs, files in os.walk(file):
            for file in files:
                if file.endswith(".xml"):
                    with open(os.path.join(root, file)) as f:
                        for index, line in enumerate(f):
                            if index == 0: continue
                            if "<?xml " in line:
                                xml_doc.append(xml_string)
                                xml_string = ""
                            else:
                                xml_string += line

        print("Length of docs: ", len(xml_doc))

        patents = []
        for doc in xml_doc:
            index = 0
            try:
                myroot = ET.fromstring(doc)
                title = ""
                abstract = ""
                doc_num = ""
                for x in myroot.findall('.//invention-title'):
                    title = x.text
                for x in myroot.findall('.//abstract/p'):
                    abstract = x.text
                for x in myroot.findall('.//doc-number'):
                    doc_num = x.text
                item_data = []
                item_data.append(title)
                item_data.append(abstract)
                item_data.append(doc_num)
                if item_data[0] == "NO": #skip meaning less title
                    continue
                if item_data[0] is "" or item_data[0] is None or item_data[1] is "" or item_data[1] is None:
                    continue
                patents.append(item_data)
            except:
                continue  # skip wrong format data

        print("Length of docs after removing data with wrong format: ", len(patents))

        df = pd.DataFrame(patents, columns=['Title', 'Abstract', 'Patent Number'])
        # combine title and abstract to do analysis
        df['title_abstract'] = df.Title + ' ' + df.Abstract
        df.tail()

        print("Training Model")
        cv = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF,
                             ngram_range=(1, 1),
                             tokenizer=tokenize, lowercase=True)
        count = cv.fit_transform(df.title_abstract.values.astype(str))
        terms = cv.get_feature_names()
        vectorizer = TfidfTransformer(use_idf=True,
                                     smooth_idf=True,
                                     sublinear_tf=True)
        doc_term_matrix = vectorizer.fit_transform(count)
        tf_idf_norm = normalize(doc_term_matrix)
        tf_idf_array = tf_idf_norm.toarray()
        tf_idf_array.shape

        # fit knn model
        knn = NearestNeighbors(n_neighbors=N_NEIGHBORS)
        knn.fit(tf_idf_array)

        print("Saving Model")
        #save model
        filename = 'finalized_knn.pkl'
        joblib.dump(knn, filename)
        filename = 'finalized_vectorizer.pkl'
        joblib.dump(cv, filename)
        filename = 'finalized_tfidf.pkl'
        joblib.dump(vectorizer, filename)
        df.to_csv(r'imported_data.csv')


if __name__ == "__main__":
    model = KNNModel()
    model.train()
    print("Train Completed")
