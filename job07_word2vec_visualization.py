import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus'] = False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
key_word = '재미있다'
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
print(sim_word)
print(list(embedding_model.wv.index_to_key)) #학습된 단어의 목록
print(len(embedding_model.wv.index_to_key)) #학습된 단어의 총 개수
vectors = []
labels = []

for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv[label])
print(vectors[0])
print(len(vectors[0]))

df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

tsne_model = TSNE(perplexity=9, n_components=2, init='pca', n_iter=2500)
new_value = tsne_model.fit_transform(df_vectors)

# TSNE를 사용하여 100차원 벡터를 2차원으로 축소
# perplexity=9 → 데이터의 밀도를 고려하는 매개변수
# n_components=2 → 2차원으로 변환
# init='pca' → 초기값을 PCA 방식으로 설정

# PCA의 주요 목적
# 1.차원 축소
#   고차원 데이터를 2D 또는 3D로 줄여 시각화 및 연산 효율성 개선
# 2.노이즈 제거
#   데이터의 주요 패턴을 유지하며 노이즈를 줄임
# 3.특징 추출
#   중요한 특성(주성분)을 추출하여 모델 성능 향상
# n_iter=2500 → 2500번 반복하여 학습

df_xy = pd.DataFrame({'words':labels, 'x':new_value[:, 0],
                      'y':new_value[:, 1]})
df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)
print(df_xy)
print(df_xy.shape)

plt.figure(figsize=(8, 8))
plt.scatter(0, 0, s=1500, marker='*')

for i in range(len(df_xy)):
    a = df_xy.loc[[i, 10]]
    plt.plot(a.x, a.y, '-D', linewidth=1)
    plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()


















