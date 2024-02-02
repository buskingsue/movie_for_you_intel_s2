import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NanumBarunGothic')

df = pd.read_csv('./cleaned_one_review.csv')
words = df.iloc[1906, 1].split()
print(words)

worddict = collections.Counter(words)
worddict = dict(worddict)
print(worddict)

wordcloud_img = WordCloud(
    background_color='white', max_words=2000, font_path=font_path
    ).generate_from_frequencies(worddict)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()



















