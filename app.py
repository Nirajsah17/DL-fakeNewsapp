from flask import Flask,render_template,jsonify,request,url_for,redirect
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras


voc_size = 10000
def removes(news):

  corpus=[]
  review = re.sub('[^a-zA-Z]', ' ', news)
  review=review.lower()
  review=review.split()

  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)
  return corpus

def clean(corpus):
  onehot_repr=[one_hot(words,voc_size)for words in corpus]
  sent_length=20
  embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
  finalnew = np.array(embedded_docs)
  return finalnew

model = keras.models.load_model('model3.h5')

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('home.html')
@app.route('/predict',methods=['post'])
def predict():
    newsp = request.form['search']
    removed_doc = removes(newsp)
    feature_news = clean(removed_doc)
    p = model.predict_classes(feature_news)
    def pre(p):
      if p==0:
        return '------> News is Fake !!!!! share with Friends :'
      else:
        return "------> News is True ,share with friends"
    prediction = pre(p)
    return render_template('predict.html', prediction_text=prediction,news_txt = newsp)
@app.route('/news')
def news():
    return render_template('news.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')



if __name__=='__main__':
    app.run(debug=True)