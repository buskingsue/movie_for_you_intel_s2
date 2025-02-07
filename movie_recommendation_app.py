#1. 패키지 및 모듈 불러오기
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle #파일 이름과 임포트 패키지 이름이 같으면 안됨
from PyQt5.QtCore import QStringListModel

#2. UI 파일 로드
form_window = uic.loadUiType('./movie_recommendation.ui')[0]

#3. GUI 클래스 정의
class Exam(QWidget, form_window):

#4. 클래스 생성자 (__init__)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
        with open('./models/tfidf_vectorizer.pkl', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
        self.df_reviews = pd.read_csv('./cleaned_reviews.csv')
        self.titles = list(self.df_reviews['movie_title'])
        self.titles.sort()
        for title in self.titles:
            self.comboBox.addItem(title)

#5. 자동완성 기능 추가
        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

#6. 이벤트 핸들러 연결
        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommendation.clicked.connect(self.btn_slot)
        self.le_keyword.returnPressed.connect(self.btn_slot)  # Enter 키를 누르면 btn_slot 실행
        #enter 키를 누르면 추천 버튼이 눌러짐
#7. 버튼 클릭 시 동작
    def btn_slot(self):
        key_word = self.le_keyword.text()
        if key_word in self.titles:
            recommendation = self.recommendation_by_movie_title(key_word)
        else :
            recommendation = self.recommendation_by_keyword(key_word)
        if recommendation:
            self.lbl_recommendation.setText(recommendation)

#8. 콤보박스 선택 시 동작
    def combobox_slot(self):
        title = self.comboBox.currentText()
        print(title)
        recommendation = self.recommendation_by_movie_title(title)
        print('debug01')
        self.lbl_recommendation.setText(recommendation)
        print('debug02')

#9. 키워드 기반 추천
    def recommendation_by_keyword(self, key_word):
        try:
            sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
        except:
            self.lbl_recommendation.setText('제가 모르는 단어에요 ㅎㅎ')
            return 0
        words = [key_word]
        for word, _ in sim_word:
            words.append(word)
        setence = []
        count = 10
        for word in words:
            setence = setence + [word] * count
            count -= 1
        setence = ' '.join(setence)
        print(setence)
        setence_vec = self.Tfidf.transform([setence])
        cosine_sim = linear_kernel(setence_vec, self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

#10. 영화 제목 기반 추천
    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['movie_title'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation

#11. 추천 영화 리스트 반환
    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[1:11]

#12. GUI app 실행 코드
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
