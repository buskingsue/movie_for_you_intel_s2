import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  # 파일 이름과 패키지 이름이 같으면 안됨
from konlpy.tag import Okt

# Okt 형태소 분석기 초기화
okt = Okt()

# 토크나이저 함수 정의
def tokenizer(text):
    if pd.isna(text):
        return []  # 빈 리스트 반환
    return okt.morphs(str(text))  # 문자열 변환 후 형태소 분석

# 1. CSV 파일 읽기
df = pd.read_csv('cleaned_reviews.csv', encoding='utf-8')  # 인코딩 설정 (필요시 'cp949' 등 변경)
text_data = df['review'].dropna().tolist()  # 결측치 제거 후 리스트 변환

# 2. TF-IDF 벡터화기 생성 및 변환
vectorizer = TfidfVectorizer(
    max_features=1000,
    tokenizer=tokenizer  # 커스텀 토크나이저 사용 (token_pattern 제거)
)

X = vectorizer.fit_transform(text_data)

# 3. 벡터화기 저장
os.makedirs('models', exist_ok=True)  # 폴더가 없으면 생성
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 4. 변환 결과 확인
print("변환된 벡터 차원:", X.shape)
