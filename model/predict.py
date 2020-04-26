import re
import praw
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle
import os

nltk.download('stopwords')

cwd = os.getcwd()
model = pickle.load(open('/model/model.pkl', 'rb'))
vector = pickle.load(open('/model/ngrams_vectorizer.pkl', 'rb'))
flairs = ['AMA', 'AskIndia', 'Business/Finance', 'Coronavirus', 'Food', 'Non-Political', 'Photography',
          'Policy/Economy', 'Politics', 'Scheduled', 'Science/Technology', 'Sports']

reddit = praw.Reddit(client_id='-mb7llu03-AJvg', client_secret='YFPwnS8jjILemmCRNvzM3SAZKT4', user_agent='redditflare',
                     username='ashishsalunkhe', password='red@2722ash')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    if type(text) == np.nan:
        return np.nan
    text = str(text)
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def data_prep(res_url):
    posts = reddit.submission(url=str(res_url))
    res_data = {"title": str(posts.title), "title_u": str(posts.title), "content": str(posts.selftext)}

    posts.comments.replace_more(limit=5)
    combined_comments = " "
    for comment in posts.comments:
        combined_comments += " " + comment.body
    res_data["combined_comments"] = str(combined_comments)
    res_data['title'] = clean_text(str(res_data['title']))
    res_data['content'] = clean_text(str(res_data['content']))
    res_data['combined_comments'] = clean_text(str(res_data['combined_comments']))
    return res_data


def pred(pred_url):
    features = data_prep(pred_url)
    final_features = str(features['title']) + str(features['selftext']) + str(features['combined_comments'])
    data = pd.DataFrame({"content": [final_features]})
    final_feature = vector.fit_transform(data.content).toarray()
    res = flairs[int(model.predict(final_feature))]
    return {"result": res, "title": str(features['title_u'])}
