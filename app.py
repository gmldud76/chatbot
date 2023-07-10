## 라이브러리
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from PyPDF2 import PdfReader

import streamlit as st
from streamlit_chat import message

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import re
from translate import Translator
import base64


##  function
# 번역
def translate_text(text, target_language='ko'):
    translator = Translator(to_lang=target_language)
    translated = translator.translate(text)
    return translated

# 영어 확인
def has_english_word(text):
    return bool(re.search(r'[a-zA-Z]', text))

# 답변 자르기
def cut_off_response(response, max_response_length):
    if len(response) >= max_response_length:
        cut_off_index = response.rfind('.', 0, max_response_length)
        if cut_off_index != -1:
            response = response[:cut_off_index + 1]
    return response

# 백그라운드 이미지
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# 대화 내용 초기화
def on_btn_click():
    del st.session_state['generated'][:]
    del st.session_state['past'][:]
    del st.session_state['chat_history'][:]


def main() :
    # env 로드  
    load_dotenv()

    # 홈페이지 이름
    st.set_page_config(page_title = '영천시 관광 가이드 홈페이지')
    
    # 백그라운드 사진
    # add_bg_from_local('배경.png')  

    # 헤드라인
    col1, col2 = st.columns([1,30])
    with col1:
        st.image('yc_chr.png', width = 120)
    with col2:
        st.markdown("<h1 style='text-align: center; color: black;'>영천시 관광 가이드 챗봇</h1>", unsafe_allow_html=True)
        st.markdown('')
    
    ## 대화내용 저장 공간
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if "chat_history " not in st.session_state:
        st.session_state['chat_history'] = []
                
    response_container = st.container()

    container = st.container()

    # 파일 업로드
    pdf = "영천시데이터.pdf"
    #pdf = "Yeoungchung.pdf"
    

    # 텍스트 추출
    if pdf is not None :
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages : 
            text += page.extract_text()

        # 청크로 분활
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1200,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # 텍스트 임베딩 
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

    
   
   # 질문창 및 답변 생성
    with container:
        with st.form(key='form', clear_on_submit=True):
            user_question = st.text_area('영천시 관광에 대해 물어보세요.:sunglasses:','', key="user_question", height=100)
            submit_button = st.form_submit_button(label='전송')

        if submit_button and user_question : 
            llm = OpenAI(model_name= 'gpt-3.5-turbo-16k') # gpt-3.5-turbo-16k 
    
            with get_openai_callback() as cb :  
                with st.spinner("Thinking...") : 
                     
                    st.session_state['chat_history'].append({"role": "user", "content": user_question})
                    
                    # 메모리에 내용 저장
                    memory = ConversationBufferMemory(memory_key = 'chat_history',
                                                          return_messages = True)
                    # 메모리 및 llm, 텍스트 임베딩 연결
                    chain_memory = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=knowledge_base.as_retriever(),
                                            memory=memory )
                    # 답변
                    response = chain_memory.run(question = user_question)

                    # 답변에 영어가 있으면 번역
                    if has_english_word(response):
                         response = translate_text(response, target_language='ko')

                    st.session_state['chat_history'].append({"role": "Ai", "content": response})
                    print(cb)
                    
    

            
            st.session_state['past'].append(user_question)
            st.session_state['generated'].append(response)
        


    # 대화 내용 출력
    with response_container : 
        con = st.container()    
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            if i == 0 :
                con.caption('대화내용')
    
    # 대화 내용 다운로드
    download_str = []
    for i in range(len(st.session_state['generated'])):
        download_str.append('질문: '+ st.session_state["past"][i])
        download_str.append('답변: '+ st.session_state["generated"][i])
    download_str = '\n'.join(download_str)
    if download_str:
        st.sidebar.download_button('Download', download_str)

    # 사이드바 내용
    st.sidebar.button("New Chat", on_click=on_btn_click, type='primary')
    st.sidebar.write("사용법\n\n질문을 입력하고 '전송'버튼을 누르시면 질문에 대한 답변이 나옵니다.\n\n새로운 대화를 하고 싶으시거나 질문에 대한 대답에 문제가 발생하시면 위에 있는 'New Chat'버튼을 눌러 다시 질문해주시면 감사하겠습니다.\n\n대화내용을 다운로드하고 싶으시면 질문 후 'New Chat' 위에 생성되는'Download'버튼을 클릭하시면 txt 파일로 다운로드 가능합니다.  ")






if __name__ == '__main__' :
    main()

