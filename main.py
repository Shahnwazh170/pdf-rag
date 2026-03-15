import logging
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure the root logger to output INFO level messages and higher to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

FILE_NAME = "ShahnwazHusain_Resume.pdf"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
DB_PATH = "./vectorstore"

api_key = os.getenv("GOOGLE_API_KEY")
embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7, max_tokens=None, max_retries=2, timeout=None, )


def get_vector_db():
    if os.path.exists(DB_PATH):
        logging.info("Loading existing database")
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    else:
        loader = PyPDFLoader(FILE_NAME)
        document = loader.load()
        # Create Chunks from TEXT
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, length_function=len)
        chunks = splitter.split_documents(document)
        logging.info("Starting embedding and storing the chunks")
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=DB_PATH)
        logging.info("Embedding and storing completed")
        return vector_db


def main():
    vector_db = get_vector_db()
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    # logging.info("Checking query")
    # question = "tell me about your experience in python"
    while True:
        user = input("User: ")
        if user.lower() == "exit":
            break
        else:
            context = vector_retriever.invoke(user)
            context_content = "\n\n".join([doc.page_content for doc in context])
            logging.info(f'Found result : {context_content}')
            prompt = f"""Use only the context below to answer the question. If the answer is not in the context, 
say "I don't know".
Context: {context_content}
Question: {user}"""
            result = llm.invoke(prompt)
            print(f"Assistant: {result.content}")

    # for i, doc in enumerate(data):
    #     logging.info(f'[{i + 1}] (Page {doc.metadata.get('page', '??')}): {doc.page_content[:200]}')

    # Convert PDF to TEXT
    # text = pdf_to_txt(FILE_NAME)
    # print(text)

    # Create Chunks from TEXT
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    # chunks = splitter.split_text(text)

    # Create embeddings and store in DB
    # create_embeddings_from_document()


# def pdf_to_txt(filename):
#     text = ""
#     loader = PyPDFLoader(filename)
#     pages = loader.load()
#
#     for i, page in enumerate(pages):
#         print(f'Reading page: {i + 1}')
#         text += page.page_content[:] + "\n"
#
#     return text


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
