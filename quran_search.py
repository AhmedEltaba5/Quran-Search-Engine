import numpy as np
import pandas as pd
import re
import string
import pickle
from typing import Dict, List, Union

import pyarabic.araby as araby
from nltk.corpus import stopwords # arabic stopwords
import arabicstopwords.arabicstopwords as stp # arabic stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import ArabicStemmer # Arabic Stemmer gets root word
import qalsadi.lemmatizer

st = ArabicStemmer()
lemmer = qalsadi.lemmatizer.Lemmatizer()

class SearchEngine:
    def __init__(self):
        self.df = pd.read_csv('data/clean_df.csv')
        self.vectorizer = pickle.load(open('model/tfidf_vectorizer.pickle', 'rb'))
        self.corpus_vectorized = pickle.load(open('model/corpus_vectorized.pickle', 'rb'))
        self.stopwordlist = self._get_stopwords()

    def _get_stopwords(self) -> List:
        """
            helper function to get arabi stopwords        
        """
        stopwordlist = set(list(stp.stopwords_list()) + stopwords.words('arabic'))
        stopwordlist = [self.normalize_chars(word) for word in stopwordlist]

        return stopwordlist

    def normalize_chars(self, txt: str) -> str:
        """
            helper function to normalize arabic characters
        """
        txt = re.sub("[إأٱآا]", "ا", txt)
        txt = re.sub("ى", "ي", txt)
        txt = re.sub("ة", "ه", txt)

        return txt

    def clean_txt(self, txt: str) -> str:
        """
            helper function for text cleaning & preparation
        """
        # remove tashkeel & tatweel
        txt = araby.strip_diacritics(txt)
        txt = araby.strip_tatweel(txt)
        # normalize chars
        txt = self.normalize_chars(txt)
        # remove stopwords & punctuation
        txt = ' '.join([token.translate(str.maketrans('','',string.punctuation)) for token in txt.split(' ')\
                        if token not in self.stopwordlist])
        # lemmatizer
        txt_lemmatized = ' '.join([lemmer.lemmatize(token) for token in txt.split(' ')])
        out_txt = txt+" "+txt_lemmatized

        return out_txt

    def get_best_results(self, df_quran: pd.DataFrame, scores_array: np.array, top_n: int) -> List[Dict]:
        """
            retrieve the top_n ayah with the highest scores and show them
        """
        results = []
        sorted_indices = scores_array.argsort()[::-1]
        for position, idx in enumerate(sorted_indices[:top_n]):
            row_pred = {}
            score = scores_array[idx]
            if score > 0:
                row = df_quran.iloc[idx]
                row_pred['ayah_txt'] = row["ayah_txt"]
                row_pred['ayah_num'] = row["ayah_num"]
                row_pred['surah_name'] = row["surah_name"]
                results.append(row_pred)
        return results

    def run_search(self, query: str) -> List[Dict]:
        """
            run tfidf to get search results
        """
        query = self.clean_txt(query)
        query_vectorized = self.vectorizer.transform([query])
        scores = query_vectorized.dot(self.corpus_vectorized.transpose())
        scores_array = scores.toarray()[0]
        return self.get_best_results(self.df, scores_array, top_n=20)
