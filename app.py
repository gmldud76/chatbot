## 라이브러리
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from PyPDF2 import PdfReader

import streamlit as st
import streamlit_chat as stc
from streamlit_chat import message


from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.callbacks import get_openai_callback

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

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
    del st.session_state['generated']
    del st.session_state['past']
    del st.session_state['chat_history']


def main() :
    # env 로드  
    load_dotenv()
    st.set_page_config(page_title = 'Ask your PDF')
    
    # 백그라운드 사진
    # add_bg_from_local('background.jpg')  

    # 헤드라인
    st.markdown("<h1 style='text-align: center; color: black;'>영천시 관광 가이드 챗봇</h1>", unsafe_allow_html=True)

    ## 대화내용 저장 공간
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if "chat_history " not in st.session_state:
        st.session_state['chat_history'] = []
                
    response_container = st.container()

    container = st.container()

    # 사이드 바
    # with st.sidebar :
    #     st.button("New Chat", on_click=on_btn_click, type='primary')
    #     st.write("사용법")

    # 파일 업로드
    pdf = "영천시데이터.pdf"
    

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

        
    # 대화창 생성
    col1, col2 = st.columns([1,3])
    with col1:
        st.image('yc_chr.png', width = 120)

    #col3, col4 = st.columns([1,8])
    #with col3:
        #st.image('yc_chr.png', width = 120)
    # with col4:
    #     with st.form('form', clear_on_submit=True):
    #         user_question = st.text_input('영천시 관광에 대해 물어보세요.:sunglasses:','', key="user_question")
    #         submit_button = st.form_submit_button('전송')

    

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

                    st.session_state['chat_history'].append({"role": "user", "content": response})
                    print(cb)
            
            st.session_state['past'].append(user_question)
            st.session_state['generated'].append(response)
        


    # 대화 내용 출력
    if st.session_state['chat_history'] : 
        with response_container : 
            con = st.container()    
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                if i == 0 :
                    con.caption('대화내용')
            

   
    download_str = []
    for msg in st.session_state['chat_history'][0:]:
        download_str.append(msg['content'])
    download_str = '\n'.join(download_str)
    if download_str:
        st.sidebar.download_button('Download', download_str)

  
    st.sidebar.button("New Chat", on_click=on_btn_click, type='primary')
    st.sidebar.write("사용법")


if __name__ == '__main__' :
    main()

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("드라마 더 글로리 출연진이 누구야?")
