from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# from langchain.chains import ConversationalRetrievalChain

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_chat import message # 챗봇만드는 import 

from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAILangChainMemory

# streamlit run app.py

col1, col2 = st.columns([2,8])
with col1:
    st.image('yc_chr.png', width=150)
with col2:
    st.title('영천시 관광 챗봇')
st.markdown('')

def main() :
    load_dotenv()

    #st.set_page_config(page_title = 'Ask your PDF')
    #st.header('영천시 관광 챗봇 -영광이-')

    # upload file
    # pdf = st.file_uploader('Upload your PDF', type = 'pdf')
    pdf= "Yeongcheon.pdf"

    # extract the text
    if pdf is not None :
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages : 
            text += page.extract_text()

        # split into chnuks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 500,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)


        # show your question
        col3, col4 = st.columns([1,8])
        with col3:
            st.image('yc_chr.png', width = 100)
        with col4:
            placeholder_text = '예시1: 영천시 관광 추천해줘.'
            query = st.text_area("영천시 관광에 대해 물어보세요.:sunglasses: SEND 버튼 또는 Ctrl+Enter를 누르면 답변이 생성됩니다.", "", placeholder=placeholder_text)
        
        #user_question = st.text_input("영천시 관광에 대해 물어봐", key="user_question")
        
        if query :
            docs = knowledge_base.similarity_search(query)
            
            llm = OpenAI(model_name= 'gpt-3.5-turbo')

            # template = """If it is not a question about the pdf content, say 'Our chatbot can only answer questions about PDFs.'
            #     Question: {user_question}
            #     Answer: 'Our chatbot can only answer questions about PDFs.'"""
            # prompt = PromptTemplate(template=template, input_variables=['user_question'])

            # chain = load_qa_chain(prompt = prompt, llm = llm, chain_type = "stuff", document_variable_name="user_question")

            chain = load_qa_chain(llm = llm, chain_type = "stuff")
                # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), knowledge_base.as_retriever())
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            st.session_state.messages.append(HumanMessage(content = query))

            with st.spinner("Thinking...") : 
                with get_openai_callback() as cb : 
                    if 'code' in query.lower():
                        response = "We are a chatbot that answers about tourism in Yeongcheon."
                    else:
                        response = chain.run(input_documents = docs, question = query)
                    
                    print(cb)

                # st.write(response) ## 이부분 없에면 챗봇형식이 될듯.
                # response 를 기억하는 형식으로 만들고 싶은데 지금 아는 방식으로는
                # similar 기능이 사라져서 형식에 어긋나 버림..
            st.session_state.messages.append(AIMessage(content=response))
            
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[0:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')



if __name__ == '__main__' :
    main()






# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import OpenAI

# llm = OpenAI(temperature=0)

# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("드라마 더 글로리 출연진이 누구야?")
