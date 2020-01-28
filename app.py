from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import requests
from readability import Document
import re
from unicodedata import normalize


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('../webapp/revised_rating_data')
    lemmatized = df['lemmatized'].tolist()
    X = df['lemmatized']
    y = df['point_non-bad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    tvec = TfidfVectorizer(stop_words='english')
    tvec.fit(X_train.values.astype('U'))
    X_train = tvec.transform(X_train.values.astype('U'))
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    if request.method == 'POST':
        message = request.form['message']
        data = message
        response = requests.get(data)
        doc = Document(response.text)
        full_text = doc.summary(html_partial=True)
        full_text = full_text.replace(r"\n", " ")
        full_text = full_text.replace(r"\t", " ")
        full_text = full_text.replace(r"/", " ")
        full_text = full_text.replace(r"<p>", " ")
        full_text = normalize('NFKD', full_text)
        full_text = full_text.split('< p>')
        TAG_RE = re.compile(r'<[^>][^>]+>')
        
        def remove_tags(text):
            return TAG_RE.sub(' ', text)
        
        term_text = list(map(remove_tags, full_text))
        term_frame = pd.DataFrame(np.array(term_text), columns = ['quoteText'])
        
        def text_to_words(titletext):
            letters_only = re.sub("[^a-zA-Z]", " ", titletext)
            words = letters_only.lower().split()
            lemmatizer = WordNetLemmatizer()
            tokens_lem = [lemmatizer.lemmatize(i) for i in words]
            return(' '.join(tokens_lem))
        
        lemm_text=[]
        for text in term_frame['quoteText']:
            lemm_text.append(text_to_words(text))
            
        vect = tvec.transform(lemm_text).toarray()
        prediction = pd.DataFrame(lr.predict_proba(vect), columns=['warning','non-warning'])
        results = pd.merge(term_frame, prediction, left_index=True, right_index=True)
        my_prediction = results["warning"].mean()
        return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)