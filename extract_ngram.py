import json
import os
from os.path import join
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from scipy import io
import pickle


def clean_string(value):
    value = value.lower()
    value = value.encode('ascii', errors = 'replace')
    value = value.translate(None, string.punctuation)
    return value

#extract 1grams tfidf features
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


if __name__ == '__main__' :
    directory = "articles"
    i = 0
    articles = []
    tags = []
    topics_dict = {}
    for filename in os.listdir(directory):
        topics_dict[i] = filename.split('.')[0]
        if filename.endswith(".json"):
            with open(join(directory, filename), 'rb') as f:
                json_data = f.read()
                data = json.loads(json_data)
                for key, value in data.items():
                    value = clean_string(value)
                    articles.append(value)
                    tags.append(i)
            i += 1

    stemmer = PorterStemmer()
    pickle.dump(tags, open('tags.pk', 'w'))
    # initialize the TfidfVectorizer 
    tfidf_vect = TfidfVectorizer(tokenizer=tokenize, stop_words = 'english',\
                         max_df = 0.5, min_df = 20)
    dtm = tfidf_vect.fit_transform(articles)
    io.mmwrite("dtm_1gram.mtx", dtm)
    # initialize the TfidfVectorizer 
    tfidf_vect = TfidfVectorizer(tokenizer=tokenize, ngram_range = (1,2),\
                         stop_words = 'english', max_df = 0.5, min_df = 20)
    # generate tfidf matrix for 1, 2 grams
    dtm= tfidf_vect.fit_transform(articles)
    io.mmwrite("dtm_2gram.mtx", dtm)
    
