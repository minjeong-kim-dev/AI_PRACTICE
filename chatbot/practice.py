# 기본 챗봇을 Streamlit으로 구현한다.
# 고도화) 음성 대화 기능을 추가해본다. (TTS: 챗봇 응답을 음성으로 출력)
# 고도화) 음성 대화 기능을 추가해본다. (STT)

from openai import OpenAI
from dotenv import load_dotenv      # .env 파일의 환경변수 로드
import os                           # 환경변수 접근용
import streamlit as st              # streamlit 라이브러리

load_dotenv()                       # 현재 위치의 .env를 읽어와 환경변수로 등록
api_key = os.getenv('openai_key')   # .env의 openai_key 값을 가져옴
client = OpenAI(api_key=api_key)

# ===============================
# Streamlit 기본 화면 설정
# ===============================
# 웹 페이지 제목과 파비콘 설정
st.set_page_config(page_title='맹PT', page_icon='🪨')

# 페이지 상단 제목 출력
st.title('🪨맹PT')

# ===============================
# 세션 상태(Session State) 초기화
# ===============================
# 세션에 messages가 없다면(첫 실행이라면)
if 'messages' not in st.session_state:
    # 대화 기록을 저장할 리스트 초기화
    # system 메시지는 챗봇의 기본 성격/설정 역할
    st.session_state.messages = [
        {'role' : 'system', 'content' : """
                                        당신은 애니메이션 '짱구는 못말려'의 등장인물 맹구입니다.
                                        사용자와 대화할 때 맹구의 말투와 성격으로 직접 말합니다.
                                        말투는 느리고 단순하며, 말수가 적습니다.
                                        상담사나 도우미처럼 설명하지 말고, 캐릭터로만 대답하세요.
                                        질문을 던지더라도 맹구답게 짧게 말합니다.
                                        맹구는 사용자가 먼저 말을 걸기 전까지는 사용자에게 말을 먼저 걸지 않습니다. 
                                        """}
    ]

# ===============================
# 이전 대화 기록 화면에 출력
# ===============================
# 저장된 모든 메시지를 순회하며 채팅 UI로 출력
for msg in st.session_state.messages:
    if msg['role'] != 'system':                 # system 메시지는 화면에 출력하지 않음
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# ===============================
# 사용자 텍스트 입력 처리
# ===============================
# 하단 입력창 생성
user_text = st.chat_input('대화를 종료하고 싶다면 "돌"을 입력해주세요.')

# 사용자가 입력했다면
if user_text:
    # 사용자 메시지를 대화 기록에 저장
    st.session_state.messages.append({'role' : 'user', 'content' : user_text})
    # 사용자 메시지를 화면에 출력
    with st.chat_message('user'):
        st.markdown(user_text)

    # ===============================
    # 종료 의사 확인
    # ===============================
    if user_text == "돌":
        with st.chat_message('assistant'):
            st.markdown('음… 이제 그만할까. 다음에 또 얘기하자.')
        
        # 종료 메시지도 대화 기록에 저장
        st.session_state.messages.append(
            {'role' : 'assistant', 'content' : '음… 이제 그만할까. 다음에 또 얘기하자.'}
        )

        # GPT 호출을 하지 않도록 여기서 종료
        st.stop()

# ===============================
# Assistant 응답 생성 및 출력
# ===============================
with st.chat_message('assistant'):
    # GPT 응답 생성 중 로딩 표시
    with st.spinner('🪨🪨🪨'):
        # OpenAI Chat Completion 호출
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=st.session_state.messages, # 이전 대화를 기억하는 곳 
            temperature=1,
            top_p=1
        )

        # Assistant의 응답 텍스트 추출
        assistant_text = response.choices[0].message.content

    # Assistant 텍스트를 화면에 출력
    st.markdown(assistant_text)

    # ===============================
    # TTS (Text To Speech)
    # ===============================
    # 음성 파일 저장 경로 설정
    speech_path = 'assistant.mp3'

    # Assistant 응답 텍스트를 음성으로 변환
    with client.audio.speech.with_streaming_response.create(
        model='gpt-4o-mini-tts',    # TTS 모델
        voice='onyx',               # 음성 스타일
        input=assistant_text        # 음성으로 변환할 텍스트
    ) as resp:
        # 스트리밍으로 받은 음성을 MP3 파일로 저장
        resp.stream_to_file(speech_path)
    
    # ===============================
    # 음성 재생
    # ===============================
    # 생성된 음성 파일을 Streamlit에서 재생
    st.audio(speech_path)

# ===============================
# Assistant 응답을 대화 기록에 저장
# ===============================
# 다음 대화에서 맥락으로 사용되도록 저장
st.session_state.messages.append({'role' : 'assistant', 'content' : assistant_text})   



