import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- Global Configurations (تنظیمات سراسری) ---
FILE_NAME = "ebook.pdf"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# کلید API Groq (باید با کلید واقعی جایگزین شود)
GROQ_API_KEY_VALUE = "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" 
# -----------------------------------------------

# --- توابع Core RAG ---

def load_and_chunk_document(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

@st.cache_resource
def create_vector_database(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH 
    )
    vector_db.persist()
    
    return vector_db

@st.cache_resource
def load_or_create_db():
    if GROQ_API_KEY_VALUE == "YOUR_GROQ_API_KEY_HERE":
        st.error("خطا: کلید Groq API تنظیم نشده است. لطفاً آن را در فایل app.py جایگزین کنید.")
        st.stop()
    
    if os.path.exists(CHROMA_DB_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        st.success("پایگاه داده برداری موجود بارگذاری شد.")
    else:
        with st.spinner("پایگاه داده برداری در حال ساخت است... (فاز ۱ و ۲)"):
            all_chunks = load_and_chunk_document(FILE_NAME)
            if not all_chunks:
                st.error(f"خطا در بارگذاری فایل {FILE_NAME}. مطمئن شوید فایل در دایرکتوری موجود است.")
                st.stop()
            vector_db = create_vector_database(all_chunks)
        st.success("پایگاه داده با موفقیت ساخته شد.")
        
    return vector_db

def create_rag_chain(vector_db):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        temperature=0,
        groq_api_key=GROQ_API_KEY_VALUE
    ) 
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    template = """شما یک دستیار متخصص در مقررات ملی ساختمان ایران (مبحث نوزدهم، مدیریت انرژی در ساختمان) هستید.
    لطفاً فقط بر اساس متنی که در بخش "متن مرجع" ارائه شده است، به سؤال به زبان فارسی پاسخ دهید.
    اگر پاسخ در متن مرجع وجود نداشت، به وضوح بیان کنید که اطلاعات در دسترس نیست.

    متن مرجع:
    {context}

    سؤال: {question}

    پاسخ:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- تابع تزریق CSS برای RTL ---

def inject_rtl_css():
    st.markdown("""
        <style>
            /* تنظیم جهت سراسری برای کل برنامه و چینش متن به راست */
            html, body {
                direction: rtl;
                text-align: right;
            }

            /* عنوان‌ها و هدرها */
            h1, h2, h3, h4 {
                direction: rtl;
                text-align: right;
            }
            
            /* تنظیم جهت برای محتوای اصلی Streamlit */
            .stApp {
                direction: rtl;
            }
            
            /* تنظیم جهت برای لیبل‌ها، کادر هشدار و مارک‌داون */
            .stTextInput, .stAlert, .stMarkdown, .stText {
                direction: rtl;
                text-align: right;
            }
            
            /* تنظیم جهت متن داخل کادر ورودی کاربر */
            .stTextInput > div > div > input, .stTextInput > div > div > textarea {
                direction: rtl;
                text-align: right;
            }
        </style>
    """, unsafe_allow_html=True)

# --- رابط کاربری Streamlit (هسته برنامه) ---

# تزریق CSS بلافاصله پس از شروع
inject_rtl_css()

st.set_page_config(page_title="سیستم RAG مقررات ملی ساختمان")
st.title("🤖 RAG چت‌بات: مبحث نوزدهم")
st.caption("توسعه یافته با LangChain, Groq و Streamlit")

# 1. بارگذاری یا ساخت پایگاه داده
vector_db = load_or_create_db()

# 2. ایجاد Chain RAG
rag_chain = create_rag_chain(vector_db)

# 3. فیلد ورودی کاربر (Text Input)
query = st.text_input(
    "سؤال خود را درباره مدیریت انرژی در ساختمان بپرسید:",
    placeholder="مثال: هدف اصلی مقررات مبحث نوزدهم چیست؟",
    key="user_query"
)

# 4. اجرای RAG هنگام ارسال سؤال
if query:
    with st.spinner("...در حال جستجو و تولید پاسخ"):
        try:
            # فراخوانی Chain RAG
            response = rag_chain.invoke(query)
            
            st.subheader("💡 پاسخ سیستم RAG:")
            st.markdown(response) 

        except Exception as e:
            st.error(f"خطا در اجرای Chain RAG: {e}")
            st.warning("ممکن است مشکل از اتصال اینترنت یا محدودیت‌های استفاده از Groq باشد.")