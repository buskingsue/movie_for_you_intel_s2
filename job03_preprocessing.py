import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./reviews_kinolights.csv')
df.info()

df_stopwords = pd.read_csv('./stopwords.csv')
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords + ['영화', '감독', '연출', '배우', '연기', '작품',
                         '관객', '장면', '모르다']
okt = Okt()
cleaned_sentences = []
for review in df.reviews:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem=True)
    print(tokened_review)
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])
    df_token = df_token[(df_token['class']=='Noun') |
                        (df_token['class']=='Adjective') |
                        (df_token['class']=='Verb')]
    words = []
    for word in df_token.word:
        if 1 < len(word):
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df['reviews'] = cleaned_sentences
df.dropna(inplace=True)
df.to_csv('./cleaned_reviews.csv', index=False)

print(df.head())
df.info()

df = pd.read_csv('./cleaned_reviews.csv')
df.dropna(inplace=True)
df.info()
df.to_csv('./cleaned_reviews.csv', index=False)














