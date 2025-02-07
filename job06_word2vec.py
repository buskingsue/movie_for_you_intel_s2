import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('./cleaned_reviews.csv')
df_review.info()

reviews = list(df_review.reviews) #코드 수정 리스트에서 .reviews로 바꿈
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100, window=4,
            min_count=20, workers=4, epochs=100, sg=1)
# Word2Vec 모델을 학습
# tokens: 학습할 문장 리스트
# vector_size=100: 단어 임베딩 차원을 100으로 설정
# window=4: 한 단어를 중심으로 좌우 4개의 단어까지 고려
# min_count=20: 최소 20번 이상 등장한 단어만 학습에 사용
# workers=4: CPU 코어 4개 사용
# epochs=100: 100번 반복 학습
# sg=1: Skip-gram 방식 사용 (0이면 CBOW)
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))


















