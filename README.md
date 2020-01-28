## Capstone Project - Classifying the Terms of Service 

### Problem Statement
---
Is it possible to predict concerning terms of service and categorizing them into appropriate topics - Using Natural Language Processing ?

### Data Collection
---
The analysis was done on 383 services stated in the Terms of Service; Didn't Read (ToS;DR) website - public database of crowdsourced extracts and reviews. By requesting and reading JSON API files given in its  GitHub open sourced for use, I was able to attain 2565 data points from the stated 383 services varying in respective quantity.  by scraping the [Terms of Service; Didn't Read](https://tosdr.org/#)

### Data Cleaning

Missing data was a prominent issue in the dataset that was rendered. 
Columns that a high number of missing values were of Topics and Ratings which had to be filled up manually by the most probable value with respective of the existing values and with reference to ToSdr website.

- Columns with irrelevant values were dropped 
- Columns with large number of null values were dropped
- Topics were consolidated for consistency 
- Topics with insignificant counts were reassigned to suitable prominent topics
- Topics with low data counts (below 10 counts) were dropped

2565 rows × 18 columns was reduced to 2403 rows × 8 columns

### Feature Engineering
---
Count vectoriser was used to reveal common features in 
Feature engineering using Natural Language Processing and Sentiment Analysis gave my model more predictive power.
Regarding the features that were used in the model to help on predicting the movements were:

### Preprocessing
---

Count vectoriser was used to reveal top features in terms of service or privacy policy documents 

| top features            | frequency|
|-------------------------|----------|
| information             | 1214     |
| use                     | 1155     |
| services                | 974      |
| service                 | 679      |
| content                 | 622      |
| data                    | 601      |
| account                 | 501      |
| personal                | 472      |
| terms                   | 458      |
| time                    | 417      |


The text preprocessing was done by removing english stop words and punctuation as well as lemmatizing the words and RegExpression

Both the training and testing sets were cleaned and transformed with the same pipeline order:

*Cleaning:*
- NLTK RegeTokenizer was used to separate the words 
- Special characters were then striped away
- Only alphabets were kept
- All words were then set to lowercase
- keep only alphabets

![Image of WordCloud](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/wordcloud.png))

The text were then transformed in a TFiDFVectorizer (tvec) from the sklearn framework. The tvec was tuned using a GridSearch method measuring on accuracy of a logistic regression. English stopwords were used with a 5,000 limit of features. Once transformed, _ classification techniques were tested to find the highest accuracy score. Their training and testing scores are listed in the table below:

![Imgur](https://i.imgur.com/BQtukgM.png)

The Random Forest was identified as the best performing for highest  score on testing and smallest difference between the training and testing score. Predictions made from the Random Forest were then sent to 


Sklearn is used to fit different models.  Multinomial Naive Bayes classifier and Counter Vectorizer for preprocessing were chosen for the end model.
The reason for choosing Counter Vecctorizer is that the model should be interpretable. With Counter Vectorizer, the number of times a word occurs in a document is counted and used for prediction. It is easy to look at the word frequency of the test data.

|                         | CVect/MNB |
|-------------------------|-----------|
| Train Score Accuracy    | 0.8973    |
| Test Score Accuracy     | 0.8906    |
| Test Score ROC_AUC      | 0.8979    |
| Test Sensitivity Score  | 0.9369    |
| Train Specificity Score | 0.8588    |

### Modeling
---
The classifier chosen to do the analysis and prediction of topics was a Logistic Regression for its interpretability. 
The classifier chosen to do the analysis and categorization of concerning (or warning) terms of service extract was a RandomForest Classifier for its high precision.
TF-IDF (Term Frequency - Inverse Document Frequency) was used to vectorize the text data and variables 

*Insert comparing models and results here*

Applying classification models (Logistic Regression, Random Forest, Naive Bayes etc) to predict star ratings on cameras. Both as a multi class and binary class problem. 

### Conclusion
---
The Precision results for model with only the text documents and including all previously mentioned features
can be seen in the following image:

On the other hand, % accuracy score on the testing data (given url) is a % improvement on a baseline random choice model.

Regarding Feature importance, for documents classified as a warning or non-warning the most common words
can be seen in the image below.

![Important Features - Up](./images/up.png)
![Important Features - Down](./images/down.png)

Given those results it is worth creating models for specific services as documents seem to be generic. By creating models
per service type might improve in at least a couple percentage points for certain services.

As some final thoughts I would say that it is possible, if provided with
the right tools, models like this one could be be used to make quick decision making before accepting terms of condition.

#### Check out the deployed app for Classifying Terms of Service!

[Classifying Terms of Service Website Page](https://3a1ee6a5.ngrok.io/)

Enter the terms of service or privacy policy URL of desired service to find out whether it is 'GOOD TO GO' or 'WARNING: YOUR ATTENTION REQUIRED'. The threshold is currently set to 0.8 for a stringent analysis of warning terms. This threshold can be amended according to user preference and/or business requirements

ngrok is an open-source tool that exposes local port as a public URL through SSL which one can copy from the CLI. It provides secure tunnels to localhost server. This means that localhost web server will receive the request through public URL which is very helpful to share or demo or debug integration environments where it accepts only public URL. 

### Next Steps
---
This analysis only touches the surface and it could further be improved on advanced techniques for
Natural Language Processing. Therefore some of the future steps for this work might be to:

- Develop a convolutional and recurrent neural network model with a specialized word embedding layer
- Use Sense2Vec to understand sentences and extract more information from the text
- Deciding on a threshold (warning or non-warning) for each specfic service so that the analysis could be more precise

Other sources of information could also be explored:
- Case Studies on terms of service previous legal and ethical conflict - highlight documents that might of crucial
- Sentiment analysis
