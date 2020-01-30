## Capstone Project - Classifying the Terms of Service 

### Problem Statement
---
When was the last time you read the terms & conditions before you accepted one?

According to a US Deloitte survey of 2,000 consumers, 91% of people consent to legal terms and services conditions without reading them and this situation is even worrying as the proportion is 97% if you fall between ages 18-34. 

The language is too complex and long-winded for most service users to go through thoroughly and more often, they are willing to consumers are willingly to accept terms with the assumption most companies (at its worst) will do is sell their personal data such as name, location, age group etc. to a third party that wants to advertise to them. 

Terms of Service and Pivate Policies exist to protect the company and its users from legal trouble. But a handful of hidden clauses exist to take advantage of users.

### Proposed Solution
---

To build a tool to scan a Terms of Service or Privacy Policy page to identify unfavourable terms and classify them into categories. The idea is to potentially made these terms of services easier to understand or otherwise improve awareness.

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
Count vectoriser was used to reveal common features in Feature engineering using Natural Language Processing.

Count vectoriser was used to reveal top features in terms of service or privacy policy documents 

| top features            | frequency |
|-------------------------|-----------|
| information             | 1214      |
| use                     | 1155      |
| services                | 974       |
| service                 | 679       |
| content                 | 622       |
| data                    | 601       |
| account                 | 501       |
| personal                | 472       |
| terms                   | 458       |
| time                    | 417       |

### Preprocessing
---

The text preprocessing was done by removing english stop words and punctuation as well as lemmatizing the words and RegExpression

*Cleaning:*
- NLTK RegeTokenizer was used to separate the words 
- Special characters were then striped away
- Only alphabets were kept
- All words were then set to lowercase
- Customised english stop words were removed

#### WordCloud

![Image of WordCloud](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/wordcloud.png)

In addition to visualising keywords, Wordcloud also gives a visual mapping of the most repeated words - allowed to identitfy unneccessary html tag words such as < strong >, li etc which was drawn back to data cleaning segment to be removed once again

Though WordCloud are useful in identitying keywords, it fares poorly in obtaining significance as there is a large difference in meaning overlaps of words in various topics even with common stop words removed.

#### WordCloud by Topic

![Image of WordCloud by topic](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/wordcloud_by_topics.png)


Combination of two NLP classification models were used in the prediction of the topic prediction (out of 24 topics) as well as unfavourable terms (warning terms) 

Two bag of word models: CountVectorizer and TfidfVectorizer. Best accuracy - precision results (though very small difference) were obtained using TfidfVectorizer as compared to  CountVectorizer to vectorize the text data and variables 

|             | CVect/LogRe | TVect/LogRe  |
|-------------|-------------|--------------|
| accuracy    | 0.699       | 0.704        |
| precision   | 0.701       | 0.707        |
| recall      | 0.699       | 0.704        |
| f1 score    | 0.691       | 0.679        |



### Modeling
---

#### Topic prediction 

![Topic Distribution](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/topics.png)


The lemmatized document text were then transformed in a TFiDFVectorizer (tvec) from the sklearn framework. The tvec was tuned using a GridSearch method measuring on accuracy of the classifier. The classifier chosen to do the analysis and prediction of topics was a Logistic Regression for its interpretability.  English stopwords were used with a 5,000 limit of features. Once transformed, logistic regression classifier were tested to find the training and testing score.

The Baseline Accuracy is 12.6% in predicting one out of twenty-four subject topics.

Training score is 0.981 which indicates that overfitting. This is common in NLP as the model is fitting random idiosyncrasies in the training data rather than model the true underlying trends. Thus, looking at the test score will be of better significance.

Test score is 0.727 which is a good score conveying that the model is perfoming well enough.
This score implies that out of 100 instances, the model correctly predicts the topic 72.7 % of the time.


#### Unfavorable Terms prediction  

First Feature Engineering had to be done on the Point (Rating Classification) column

From Good-Neutral-Bad to Bad-Non-Bad

![Good-Neutral-Bad](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/bad-neutral-good.png)

![Bad-Non-Bad](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/bad-non-bad.png)

The Baseline Accuracy is 51.6% in predicting unfavourable terms.

TFiDFVectorizer (tvec) was chosen for preprocessing.
As for deciding on the model, Sklearn is used to fit different models and the precision results were compared to find out the best classifier model from the sklearn framework. The tvec was then tuned using a GridSearch method measuring on accuracy of the chosen model. 

Precision will be a good measure to determine when the costs of False Positive is high. For this case, a false positive means that a term of service that is subjected to be of warning is being identified as non-warning. The service may overlook this term of service ans agress the accept them without knowing any wiser.

![Classifier](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/classifiers.png)

Comparing the models, RandomForest seems to have the highest precision of 0.840. Thus we will be going ahead with it

|             | TVec/RandomForest |
|-------------|-------------------|
| accuracy    | 0.840             | 
| precision   | 0.840             |
| recall      | 0.840             | 
| f1 score    | 0.840             | 


Confusion Matrix was computed to give a summary of prediction results on this classification problem. 

![Confusion Matrix](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/confusion_matrix.png)

The prediction model accurately predicts:
- 84% for warning terms 
- 85% for non- warning terms

#### ROC Curve  

ROC Curve shows the tradeoff between sensitivity and specificity of our model which in this case loosely means that how good the model can distinguish.

![ROC Curve](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/(ROC)%20Curve.png)

The AUC (the area under the ROC curve) seems to be large enough therefore this show that our model is doing a good job of distinguishing the positive and the negative values.

Since our model is performing well enough, we are well convinced to move forth with it.

#### Important words for warning and non-warning terms

![Important Words](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/important%20words.png)

#### Limitations

- Wrongly predicted topics and unfavourable terms classification
- Requires a certain format of a clear segmentation to be presented for better prediction and classification 

### Conclusion
---
The Precision results for model with only the text documents and including all previously mentioned features
can be seen in the following image:

The model was able to achieve an accuracy rate of 72.7 % in predicting one out of twenty-four subject topics, against a baseline of 11% and an an accuracy rate of 84% in predicting unfavourable (warning) terms, against a baseline of 51.6%


### Check out the deployed app for Classifying Terms of Service!
---

[Classifying Terms of Service Website Page](http://9f17b6f0.ngrok.io/)

Enter the terms of service or privacy policy URL of desired service to find out whether it is 'GOOD TO GO' or 'WARNING: YOUR ATTENTION REQUIRED'. The threshold is currently set to 0.8 for a stringent analysis of unfavourable terms. This threshold can be amended according to user preference and/or business requirements

ngrok is an open-source tool that exposes local port as a public URL through SSL which one can copy from the CLI. It provides secure tunnels to localhost server. This means that localhost web server will receive the request through public URL which is very helpful to share or demo or debug integration environments where it accepts only public URL. 


#### Using [Facebook Terms of Service](https://www.facebook.com/legal/terms) as an test example

![Web App Home Page](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/Web%20Application.png)
![Web App Predict Page](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/Web%20Application%20Predict%20Page.png)
![Web App Classification Page](https://github.com/RadheV/ClassifyingTermsofService/blob/master/images/Classifying%20ToS.png)


### Next Steps
---
This analysis could further be improved on advanced techniques for
Natural Language Processing. Therefore some of the future steps for this work might be to:

- Develop a convolutional and recurrent neural network model with a specialized word embedding layer
- Use Word Sense Disambiguation methods to understand sentences and extract more information from the text
- Deciding on an appropriate threshold (warning or non-warning) for each specfic service so that the analysis could be more precise

Other sources of information could also be explored:
- Case Studies on terms of service previous legal and ethical conflict - highlight documents that might of crucial
- Sentiment analysis

Other sources of information could also be explored:
- Case Studies on terms of service previous legal and ethical conflict - highlight documents that might of crucial
- Sentiment analysis
