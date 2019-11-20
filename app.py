import pickle
import re
import numpy as np
from flask import Flask, request,render_template


app = Flask(__name__)

with open('glm_pickle','rb') as f:
    model = pickle.load(f)

with open('vectorizer_pickle','rb') as t:
    vectorizer = pickle.load(t)


#creating a function to clean the corpus
def clean_the_corpus(corpus):
    '''
    clean the corpus and returned a cleaned version of corpus
    '''
    clean_corpus = []
    for num in range(len(corpus)):
        tweet = str(corpus[num]).lower()
        tweet = re.sub(r"@user"," ",tweet)
        tweet = re.sub(r"gr8","great", tweet)
        tweet = re.sub(r"allshowandnogo","all show and go", tweet)
        tweet = re.sub(r"actorslife","actors life", tweet)
        tweet = re.sub(r"don't","do not", tweet)
        tweet = re.sub(r"can't","can not", tweet)
        tweet = re.sub(r"hv","have", tweet)
        tweet = re.sub(r"ur","your", tweet)
        tweet = re.sub(r"ain't","is not", tweet)
        tweet = re.sub(r"don't","do not", tweet)
        tweet = re.sub(r"couldn't","could not", tweet)
        tweet = re.sub(r"shouldn't","should not", tweet )
        tweet = re.sub(r"won't","will not", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"it's","it is", tweet)
        tweet = re.sub(r"its"," ", tweet)
        tweet = re.sub(r"that's","that is", tweet)
        tweet = re.sub(r"where's","where is", tweet)
        tweet = re.sub(r"who's","who is", tweet)
        tweet = re.sub(r"\W"," ", tweet)
        tweet = re.sub(r"\d"," ", tweet)
        tweet = re.sub(r"[ðâï¼½³ªãºæååçæåä¹µó¾_ëìêè]"," ",tweet)
        tweet = re.sub(r"\s[a-z]\s"," ", tweet)
        tweet = re.sub(r"\s+[a-z]\s+"," ", tweet)
        tweet = re.sub(r"\s+"," ", tweet)
        clean_corpus.append(tweet)
    return clean_corpus


@app.route('/')
def index():
    
    return render_template('sentiment.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    
    #getting the comments from screen and putting it to a list
    comments = [request.form['comments']] 
    
    clean_comment = clean_the_corpus(comments)
    comment_tfidf = vectorizer.transform(clean_comment).toarray()
    sentiment = model.predict(comment_tfidf)
    
    return render_template('final.html', comments=comments,sentiment=sentiment[0])

if __name__ == "__main__":

    app.run(debug=True)