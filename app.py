import streamlit as st
import os
from dotenv import load_dotenv

# 适配新版 LangChain 的导入路径
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载环境变量 (.env 文件中的 Key)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. 页面配置
st.set_page_config(page_title="KI-Studienberatung", layout="wide")

# ================= 侧边栏 (Sidebar) =================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
        width=50)
    st.title("Einstellungen")
    st.markdown("---")

    # 引用自文档 和 的常见德语问题
    st.subheader("Häufig gestellte Fragen")

    questions = [
        "Wie viele Fehlversuche im Grundstudium sind erlaubt?",  #
        "Was passiert bei Versäumnis einer Prüfung?",  #
        "Was sind die Unterschiede zwischen KF/AM/DAR/KBI?",  #
        "Welche Anforderungen gelten für das Praxissemester?",  #
        "Wie berechnet sich die Modulnote?",  #
        "Wieviel LP hat die Bachelorarbeit?"  #
    ]

    # 创建快速提问按钮
    for q in questions:
        if st.button(q):
            st.session_state.temp_input = q


# ================= 后台逻辑 (Backend) =================
@st.cache_resource
def get_vector_store():
    """
    加载 PDF 并建立向量索引
    """
    if not api_key:
        return None

    try:
        # 确保你的 PDF 都在 data 文件夹里
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 建立索引
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Fehler beim Laden der Dokumente: {e}")
        return None


# ================= 主界面 (Main UI) =================

st.title("🎓 KI-System für Studierendenfragen")
st.markdown("""
Willkommen! Ich bin Ihr KI-Assistent für Fragen zur **Studien- und Prüfungsordnung (SPO)** und zum Studiengang **Maschinenbau & Mechatronik**.
""")

# 检查 Key 是否存在
if not api_key:
    st.error("⚠️ Kein OpenAI API Key gefunden. Bitte überprüfen Sie die .env Datei.")
    st.stop()

# 初始化向量库
with st.spinner("System wird initialisiert... Bitte warten."):
    vector_store = get_vector_store()

if vector_store:
    # 定义模型
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=api_key)

    # 德语 Prompt 模板
    prompt = ChatPromptTemplate.from_template("""
    Du bist ein hilfreicher Assistent für die Studienberatung an einer deutschen Hochschule.
    Beantworte die Frage des Studenten basierend auf dem folgenden Kontext (Auszüge aus der SPO).

    Regeln:
    1. Antworte **ausschließlich auf Deutsch**.
    2. Verwende nur Informationen aus dem Kontext. Wenn die Antwort nicht im Kontext steht, sag: "Dazu finde ich keine Informationen in der SPO."
    3. Sei präzise und nenne, wenn möglich, die relevanten Paragraphen (§) oder Abschnitte.

    <context>
    {context}
    </context>

    Frage des Studenten: {input}
    """)

    # 创建检索链
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 聊天记录状态管理
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 处理输入 (文本框 或 侧边栏按钮)
    user_input = st.chat_input("Stellen Sie Ihre Frage hier...")

    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        st.session_state.temp_input = None

    if user_input:
        # 显示用户问题
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("Suche in der SPO..."):
                try:
                    response = retrieval_chain.invoke({"input": user_input})
                    answer = response['answer']

                    st.markdown(answer)

                    # 显示来源 (Quellen)
                    with st.expander("Quellen anzeigen (Referenz)"):
                        for i, doc in enumerate(response["context"]):
                            source_page = doc.metadata.get('page', 'Unbekannt')
                            source_file = doc.metadata.get('source', 'Dokument').split('/')[-1]
                            st.markdown(f"**Quelle {i + 1}:** {source_file} (Seite {source_page})")
                            st.caption(doc.page_content[:200] + "...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")

else:
    st.error("Datenbank konnte nicht geladen werden.")