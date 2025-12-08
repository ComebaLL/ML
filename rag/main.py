import os

# Работа с эмюеддингами
from langchain_huggingface import HuggingFaceEmbeddings

# Векторное хранилище
from langchain_community.vectorstores import FAISS

# Промоты
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Цепочки
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_ollama import ChatOllama


# Работа с файлами
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# --- 1. Загрузка и подготовка данных из файла ---

print("Загрузка файла doc_gems.txt...")

try:
    # Загружаем весь текст из файла
    loader = TextLoader("doc_gem1.txt", encoding="utf-8")
    raw_documents = loader.load()


    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,  
        chunk_overlap=0   
    )
    
    documents = text_splitter.split_documents(raw_documents)
    print(f"Файл загружен. Создано {len(documents)} фрагментов (документов) для базы знаний.")

except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    documents = [] 

if not documents:
    print("Нет данных для работы. Проверьте наличие файла doc_gems.txt")
    exit()


# --- 2. Векторизация и индексация данных ---

print("Создание векторной базы (Embeddings)...")
# Инициализируем модель для создания эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Создаем векторную базу данных FAISS из ЗАГРУЖЕННЫХ ДОКУМЕНТОВ
# Обрати внимание: используем from_documents вместо from_texts
vector_store = FAISS.from_documents(documents, embeddings_model)

# Создаем объект "ретривера"
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 означает искать 3 самых похожих скилла


# --- 3. Настройка Генеративной Модели (LLM) ---
print("Подключение к Ollama...")
llm = ChatOllama(base_url="http://localhost:11434", model="smollm2:135m")


# --- 4. Определение Промпта и Цепочки RAG ---

system_text = (
    "You are a Path of Exile game expert. Answer strictly based on the provided context. "
    "If the context matches a specific skill gem, describe it using the information provided. "
    "If you don't know, say so."
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_text),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
])

# Объект для QA
# Используем стандартный RetrievalQA (обрати внимание на импорты, langchain_classic может быть устаревшим)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
    },
)

# --- Пример запроса ---
query = "What`s skill use only sword?" 

print(f"\nВопрос: {query}")
print("-" * 30)

result = qa.invoke({"query": query})

print("Ответ LLM:\n", result["result"])

print("\n" + "="*30)
print("Использованные документы (фрагменты):")
for i, doc in enumerate(result.get("source_documents", []), 1):
    # Выводим первые 200 символов документа для проверки
    preview = doc.page_content.replace('\n', ' ')[:200]
    print(f"\n[{i}] {preview}...")