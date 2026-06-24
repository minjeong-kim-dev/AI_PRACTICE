# AI_PRACTICE

머신러닝, 딥러닝, 챗봇 실습 파일 모음

---

## 📁 폴더 구조

```
AI_PRACTICE/
├── ML/
│   └── 머신러닝 실습.ipynb       # 항공편 지연 예측
├── DL/
│   └── 딥러닝 실습.ipynb         # 항공사 리뷰 추천/비추천 예측
└── chatbot/
    ├── practice.py               # 맹구 캐릭터 챗봇 (TTS)
    └── practice2.py              # 음성 AI 어시스턴트 (STT → GPT → TTS)
```

---

## ML - 항공편 지연 예측

> 특정 조건(계절, 요일, 시간대, 거리, 항공사)에서 항공편이 지연될지 예측하는 이진 분류 모델

### 데이터
- 출처: [Kaggle - Flight Delays 2015](https://www.kaggle.com/datasets/usdot/flight-delays/data)
- 전체 데이터 중 **5,000건 랜덤 샘플링**

### 사용 Features

| 컬럼 | 설명 |
|------|------|
| MONTH | 월 (1~12) |
| DAY_OF_WEEK | 요일 (1~7, 월~일) |
| HOUR | 출발 시간대 (0~23) |
| DISTANCE | 비행 거리 |
| AIRLINE | 항공사 코드 (원핫 인코딩) |

- **Target**: `DELAYED` — 출발 지연 15분 이상 시 1, 미만 시 0

### 주요 과정
1. 결측치 제거 (DEPARTURE_DELAY 누락 59건)
2. 시각화 — 요일별 / 시간대별 평균 지연 시간
3. 타겟 컬럼 생성 (15분 기준 이진 분류)
4. AIRLINE 원핫 인코딩
5. RandomForestClassifier 학습
6. GridSearchCV로 하이퍼파라미터 튜닝 (`n_estimators`, `max_depth`, `min_samples_split`)

### 결과
- 기본 모델 정확도: **79%**

### 기술 스택
`Python` `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn`

---

## DL - 항공사 리뷰 추천/비추천 예측

> 고객이 작성한 항공사 이용 후기 텍스트를 분석해 추천 여부를 예측하는 텍스트 분류 모델

### 데이터
- 출처: [Kaggle - Airline Reviews](https://www.kaggle.com/datasets/naimaasif/airline-reviews)
- 총 **23,171건** 항공사 리뷰 텍스트

### 사용 Features

| 컬럼 | 설명 |
|------|------|
| Review | 고객이 작성한 항공사 이용 후기 텍스트 |
| Recommended | 추천 여부 (yes / no) → 타겟 |

### 주요 과정
1. 텍스트 전처리 — 영어 단어 추출, 불용어(stopwords) 제거
2. 추천/비추천별 빈출 단어 분석
3. TF-IDF 벡터화 (`max_features=1000`, `ngram_range=(1,3)`)
4. LabelEncoder로 타겟 인코딩 (no→0, yes→1)
5. 다층 퍼셉트론(MLP) 구성 (1000 → 128 → 64 → 32 → 1)
6. BCELoss + Adam 옵티마이저로 50 epoch 학습
7. 손실 곡선 시각화 및 과적합 분석

### 모델 구조
```
Linear(1000 → 128) → ReLU
Linear(128 → 64)   → ReLU
Linear(64 → 32)    → ReLU
Linear(32 → 1)     → Sigmoid
```

### 결과
| 지표 | 점수 |
|------|------|
| Accuracy | **89%** |
| Macro F1 | **0.88** |

### 기술 스택
`Python` `PyTorch` `scikit-learn` `TF-IDF` `Optuna`

---

## Chatbot - 챗봇 실습

### practice.py — 맹PT (맹구 캐릭터 챗봇)

> 짱구는 못말려의 맹구 캐릭터로 대화하는 Streamlit 챗봇 (TTS 포함)

**주요 기능**
- OpenAI GPT-4.1-mini 기반 대화
- 맹구 캐릭터 시스템 프롬프트 설정
- 챗봇 응답을 TTS(gpt-4o-mini-tts)로 음성 출력
- `"돌"` 입력 시 대화 종료
- 세션 기반 대화 기록 유지

**기술 스택**  
`OpenAI API` `Streamlit` `TTS (gpt-4o-mini-tts)`

---

### practice2.py — Voice AI Assistant

> 음성으로 입력하고 음성으로 답변받는 AI 어시스턴트 (STT → Moderation → 유사도 검색 → GPT → TTS)

**처리 파이프라인**
```
🎤 음성 녹음
    ↓
📝 STT 변환 (Whisper-1)
    ↓
🔍 Moderation 검사 (omni-moderation-latest)
    ↓ (부적절한 발언 없을 경우)
📊 Embedding 유사도 검색 (text-embedding-3-small + Cosine Similarity)
    ↓
🤖 GPT 응답 생성 (gpt-4.1-mini)
    ↓
🔊 TTS 음성 출력 (tts-1)
```

**주요 기능**
- 브라우저에서 실시간 음성 녹음 (`audiorecorder`)
- 부적절한 발언 감지 시 응답 차단
- Embedding 기반 유사 질문 검색 (Cosine Similarity)
- 처리 과정 실시간 시각화 (STT 결과, Moderation 결과, 유사도 점수)
- 사이드바에서 GPT 모델 선택 가능
- 대화 초기화 기능

**기술 스택**  
`OpenAI API` `Streamlit` `Whisper` `TTS` `Embedding` `Cosine Similarity`

---

## 🛠 전체 기술 스택

| 분야 | 기술 |
|------|------|
| 언어 | Python |
| 머신러닝 | scikit-learn, pandas, numpy |
| 딥러닝 | PyTorch, Optuna |
| 챗봇 | OpenAI API (GPT, Whisper, TTS, Embedding) |
| 프론트엔드 | Streamlit |
| 시각화 | matplotlib, seaborn |
