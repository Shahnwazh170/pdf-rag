import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chain import llm_chain
from config import MAX_RETRIES, TEMPERATURE, CHAT_MODEL, EMBEDDING_MODEL, TOP_K
from vectorstore import get_vectordb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=TEMPERATURE, max_tokens=None, max_retries=MAX_RETRIES,
                             timeout=None)


def main():
    vector_db = get_vectordb(embedding)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        else:
            result = llm_chain(llm, user_input, vector_db, vector_retriever)
            print(f"Assistant: {result}")


if __name__ == '__main__':
    main()
