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
    X_class = df['lemmatized']
    y_class = df['point_non-bad']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.25, random_state=42)
    tvec_class = TfidfVectorizer(stop_words='english')
    tvec_class.fit(X_train_class.values.astype('U'))
    X_train_class = tvec_class.transform(X_train_class.values.astype('U'))
    lr_class = LogisticRegression()
    lr_class.fit(X_train_class, y_train_class)
    
    data = pd.read_csv('../webapp/revised_data')   
    X = df['lemmatized']
    y = data['topics']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
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
            
           
        vect_class = tvec_class.transform(lemm_text).toarray()
        prediction_class = pd.DataFrame(lr_class.predict_proba(vect_class), columns=['warning','non-warning'])
        
        vect = tvec.transform(lemm_text).toarray()
        prediction = pd.DataFrame(lr.predict(vect), columns =['pred_topic'])
        
        results = pd.merge(term_frame, prediction, left_index=True, right_index=True)
        results = pd.merge(results, prediction_class, left_index=True, right_index=True)
        results = results.sort_values('non-warning')
        my_prediction = results["warning"].mean()
        #results = results[results['warning'] > 0.3 ]
        topics = []
        topicIndx = []
        topicContent=[]
        for i in results['pred_topic']:
            if i not in topics:
                topics.append(i)
        for i in topics:
            topic = results[results['pred_topic'] == i]
            count = 0
            for j in topic.index:
                count +=1   
                topicContent.append(topic.quoteText[j])
                topicIndx.append(i)
        df1 = pd.DataFrame({'topic':topicIndx,
                       'content':topicContent})
        df1 = df1.replace('\n','', regex=True)
        df1 = df1.replace('<i>','', regex=True)
        df1 = df1.replace('&#13;','', regex=True)
        return render_template('result-Copy1.html', prediction = my_prediction, df1 = df1.to_html())

if __name__ == '__main__':
    app.run(debug=True)
