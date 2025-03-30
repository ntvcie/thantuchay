#https://blog.futuresmart.ai/building-a-gpt-4-chatbot-using-chatgpt-api-and-streamlit-chat
#In chatbot.py, we start by importing the required libraries and loading the API key for the OpenAI API.
import streamlit as st
from streamlit_chat import message
#from utils import get_initial_message, get_chatgpt_response, update_chat
import os
from dotenv import load_dotenv
load_dotenv()
import openai
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
import nest_asyncio
nest_asyncio.apply()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai.api_key = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
# Cấu hình Hugging Face API

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
login(HF_TOKEN)
model = "gpt-3.5-turbo" #"gpt-4", "gpt-4o"
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
st.set_page_config(
    page_title="VIELINA-AI.CNS.01",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="auto",
)
st_style = """
<style>
    .block-container {
        padding-top: 0rem;
        padding-left: 1rem;
    }
    #MainMenu {visibility: visible;}
    footer {visibility: visible !important;}
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #262730;
        color: white;
        text-align: right;
        padding: 10px;
        font-size: 14px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.2);
    }
    header {visibility: hidden;}
    .spinner-container {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1001;
    }

    .stButton > button {
        width: 100%;        
    }

</style>
"""
st.markdown(st_style, unsafe_allow_html=True)
# === Tiêu đề và thông tin bản quyền ===
st.markdown("""
<div style="font-size: 24px; color: #1abc9c; margin: 0;">
    <span class="custom-icon">📚</span>
    <span class="custom-title">VIELINA-AI.CNS.01</span>
</div>
<div style="font-size: 16px; color: #2c3e50; margin: 0; border-bottom: 1px solid #cccc;">
    <p class="custom-footer-text">© 2025 - Hệ thống giám sát cảnh báo than tự cháy</p>
</div>
""", unsafe_allow_html=True)
# Chỉ tải FAISS index một lần khi chạy ứng dụng
if "vector_store" not in st.session_state:
    with st.spinner("Đang tải dữ liệu nhúng..."):
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        st.session_state.vector_store = FAISS.load_local("embedding_faiss_index", embeddings, allow_dangerous_deserialization=True)
# Load FAISS Vectorstore
@st.cache_resource
def load_vectorstore():
    with st.spinner("Đang tải dữ liệu nhúng..."):
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        return FAISS.load_local("embedding_faiss_index", embeddings, allow_dangerous_deserialization=True)
st.session_state.vector_store = load_vectorstore()
# Hàm truy xuất dữ liệu từ FAISS
def retrieve_context(query, top_k=3):
    docs = st.session_state.vector_store.similarity_search(query, k=top_k)
    return " ".join([doc.page_content for doc in docs])

# def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
#     print("model: ", model)
#     response = client.responses.create(
#         model=model,
#         instructions="You are a helpful AI Tutor. Who anwers brief questions about AI.",
#         input=messages,
#     )
#     return  response.output_text
#Nếu nội dung trong messages quá dài, hãy tóm tắt trước khi gửi:
def summarize_text(text, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Hãy tóm tắt văn bản sau một cách ngắn gọn nhưng vẫn giữ được nội dung chính."},
                  {"role": "user", "content": text}],
        max_tokens=500
    )
    return response.choices[0].message.content
# Hàm gọi OpenAI GPT-4o với RAG
def get_chatgpt_response(messages, model="gpt-4o"):
    query = messages[-1]["content"]  # Lấy câu hỏi mới nhất từ người dùng
    context = retrieve_context(query)  # Tìm kiếm dữ liệu liên quan từ FAISS
    print(context)
    # Nếu context quá dài, tóm tắt lại trước khi chèn vào messages
    MAX_CONTEXT_LENGTH = 500  # Số từ tối đa của context
    if len(context.split()) > MAX_CONTEXT_LENGTH:
        context = summarize_text(context, model)

    messages.insert(0, {"role": "system", "content": f"Use this information: {context}"})

    # Kiểm tra tổng số token của messages
    total_tokens = sum(len(m["content"].split()) for m in messages)
    MAX_TOKENS = 16000  # Giới hạn an toàn

    # Xóa bớt các tin nhắn cũ nếu tổng số token quá lớn
    while total_tokens > MAX_TOKENS and len(messages) > 2:
        messages.pop(1)  # Xóa tin nhắn cũ nhất (giữ lại system prompt và câu hỏi gần nhất)
        total_tokens = sum(len(m["content"].split()) for m in messages)

    # Gửi request đến OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True  # Enable streaming
    )
    return response#response.choices[0].message.content


def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

#Then, we set up the Streamlit interface with a title, subheader, and a dropdown box to select the language model (GPT-3.5-turbo or GPT-4).
# st.title("Chatbot : ChatGPT and Streamlit Chat")
# st.subheader("AI Tutor:")



#We initialize the session states to store the generated messages, past queries, and the initial set of messages.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
# Tạo bố cục hai cột
col1, col2 = st.columns([1, 1])  # Cột trái chiếm 1 phần, cột phải chiếm 2 phần

with col2:
    #query = st.chat_input("Câu hỏi của bạn")
    query = st.text_input("Câu hỏi của bạn: ", key="input")
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []#get_initial_message()
    #Next, we process the user's query and generate the AI response.
    if query:
        with st.spinner("generating..."):
            messages = st.session_state['messages']
            messages = update_chat(messages, "user", query)
            response = get_chatgpt_response(messages, model)

            full_response = ""
            with st.chat_message("assistant"):
                msg_container = st.empty()
                for part in response:
                    if part.choices and part.choices[0].delta.content:
                        full_response += part.choices[0].delta.content
                        msg_container.markdown(full_response)

            update_chat(messages, "assistant", full_response)
            st.session_state['messages'] = messages

with col1:
    # Display chat history
    for msg in st.session_state['messages']:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]

        # Bỏ qua hiển thị nếu nội dung chứa "Use this information:"
        if content.startswith("Use this information:"):
            continue

        with st.chat_message(role):
            st.markdown(content)
        st.snow()
