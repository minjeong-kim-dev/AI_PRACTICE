# STT -> 부적절한 발언 검사 -> 유사도 검색 -> TTS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import base64   
from openai import OpenAI
from dotenv import load_dotenv                  # .env 파일의 환경변수 로드
import os                                       # 환경변수 접근용
import streamlit as st                          # streamlit 라이브러리
from audiorecorder import audiorecorder         # 브라우저에서 음성 녹음 위젯
# from openai_service import stt, ask_gpt, tts 

# =========================
# 환경 설정
# =========================
load_dotenv()                                   # 현재 위치의 .env를 읽어와 환경변수로 등록
api_key = os.getenv('openai_key')               # .env의 openai_key 값을 가져옴
client = OpenAI(api_key=api_key)

# =========================
# STT 함수
# =========================
def stt(audio):
    output_filepath = 'input.mp3'
    audio.export(output_filepath, format='mp3')

    with open(output_filepath, 'rb') as f:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            file = f
        )
    os.remove(output_filepath)

    return transcription.text

# =========================
# GPT 응답 함수
# =========================
def ask_gpt(messages, model):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        top_p=1,
        max_completion_tokens=4096
    ).choices[0].message.content

# =========================
# TTS 함수
# =========================
def tts(respones:str):
    filename = 'output.mp3'
    with client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='alloy',
        input=respones
    ) as resp:
        resp.stream_to_file(filename)

    with open(filename, 'rb') as f:
        data = f.read()
        b64_encoded = base64.b64encode(data).decode()

    os.remove(filename)

    return b64_encoded
    
# =========================
# Embedding 함수
# =========================
def text_to_embedding(text):

    response = client.embeddings.create(
        model='text-embedding-3-small', 
        input=text.replace('\n', ' ')
    )

    return response.data[0].embedding

# =========================
# Cosine Similarity 함수
# =========================
def review_vector_search(query, embed_df, top_n):
    query_emb = text_to_embedding(query)

    df = embed_df.copy()
    df['cos_sim'] = df['embedding'].apply(lambda emb: cosine_similarity([emb], [query_emb])[0][0])

    return df.sort_values('cos_sim', ascending=False).head(top_n)[['Content', 'cos_sim']]

# =========================
# Moderation 함수
# =========================
def get_moderation_result(input,violation_only=True):
    response = client.moderations.create(
        model='omni-moderation-latest',
        input=input
    )

    flagged = response.results[0].flagged
    categories = response.results[0].categories.model_dump()

    rows = [
        {'category': k, 'flagged': v}
        for k, v in categories.items()
    ]

    df = pd.DataFrame(rows)

    # 🔥 핵심 로직
    if violation_only:
        df = df[df['flagged'] == True]

    return flagged, df

# =========================
# Embedding 기준 데이터 (실습용)
# =========================
embed_df = pd.DataFrame({
    'Content' : [
        "환불은 어떻게 하나요?",
        "배송은 얼마나 걸리나요?",
        "회원 탈퇴 방법 알려주세요",
        "결제 오류가 발생했어요",
        "고객센터 운영 시간은 언제인가요?"
    ]
})
embed_df['embedding'] = embed_df['Content'].apply(lambda x: text_to_embedding(x))

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(
        page_title='🎙️Voice AI Assistant',
        page_icon='🎙️',
        layout='wide'
    )

    st.header('🎙️Voice AI Assistant🎙️')
    st.markdown('---')

    with st.expander('Voice Chatbot 프로그램 처리 절차', expanded=False):
        st.write(
            """
            1. 음성 녹음  
            2. STT 변환  
            3. Moderation 검사  
            4. Embedding 유사도 검색 
            5. GPT 응답 생성
            6. TTS 음성 출력
            """
        )
    
    # =========================
    # 세션 상태 초기화
    # =========================
    system_prompt = '당신은 친절한 챗봇입니다. 사용자의 질문에 50단어 이내로 간결하게 답변해주세요.'

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role' : 'system', 'content': system_prompt}]

    if 'check_reset' not in st.session_state:
        st.session_state['check_reset'] = False

    # =========================
    # 사이드바
    # =========================
    with st.sidebar:
        model = st.radio(label = 'GPT 모델', options=['gpt-4.1-mini', 'gpt-5-nano', 'gpt-5.2'], index = 0)
        print(f'{model =}')

        if st.button(label='초기화'):
            st.session_state['messages'] = [{'role' : 'system', 'content' : 'system_prompt'}]
            st.session_state['check_reset'] = True

    # =========================
    # 화면 분할
    # =========================
    col1, col2 = st.columns(2)

    # -------------------------
    # 왼쪽: 음성 입력
    # -------------------------
    with col1:
        st.subheader('🎤 사용자 입력')
        audio = audiorecorder()

        if (audio.duration_seconds > 0) and (not st.session_state['check_reset']):
            st.audio(audio.export().read())

            # STT
            query: str = stt(audio)
            print(f'{query = }')
            st.session_state['messages'].append({'role' : 'user', 'content' : query})

            # Moderation & Similarity 
            flagged, mod_df = get_moderation_result(query)
            mod_df["flagged"] = mod_df["flagged"].map(
                {True: "🚨 True", False: "--"}
            )
            st.session_state['current_mod'] = mod_df

            if flagged:
                st.error('부적절한 내용이 감지되었습니다.')

            else:
                sim_df = review_vector_search(query, embed_df, top_n=3)
                st.session_state['current_sim'] = sim_df

                # GPT
                response:str = ask_gpt(st.session_state['messages'], model)
                print(f'{response = }')
                st.session_state['messages'].append({'role' : 'assistant', 'content' : response})

                # TTS
                base64_encoded_audio : str = tts(response)
                st.markdown("### 🔊 AI 음성 응답")
                st.html(f'''
                        <audio autoplay="true">
                            <source src="data:audio/mp3;base64,{base64_encoded_audio}">
                        </audio>
                        ''')
        else:
            st.session_state['check_reset'] = False

    # -------------------------
    # 오른쪽: 처리 과정 + 채팅 로그
    # -------------------------
    with col2:
        st.subheader('🧠 처리 과정')

        with st.expander("1️⃣ STT 결과", expanded=True):
            user_msgs = [m['content'] for m in st.session_state['messages'] if m['role'] == 'user']
            if user_msgs:
                st.write(user_msgs[-1])

        with st.expander("2️⃣ Moderation 결과"):
            if 'current_mod' in st.session_state:
                if st.session_state['current_mod']['flagged'].any():
                    st.warning("⚠️ 부적절한 발언 감지")
                    st.dataframe(st.session_state['current_mod'])
                else:
                    st.success("✅ 부적절한 발언 없음")

        with st.expander("3️⃣ Embedding 유사도"):
            if 'current_sim' in st.session_state:
                st.dataframe(st.session_state['current_sim'])

    st.divider()
    st.subheader("💬 질문 / 답변")

    for msg in st.session_state['messages']:
        if msg['role'] == 'system':
            continue

        with st.chat_message(msg['role']):
            st.markdown(msg['content'])


# =========================
# 실행
# =========================
if __name__ == "__main__":
    main()