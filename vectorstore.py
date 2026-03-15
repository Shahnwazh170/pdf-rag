import logging
import os

from langchain_chroma import Chroma

from config import DB_PATH
from loader import load_document

logger = logging.getLogger(__name__)


def create_vectordb(embedding, chunks):
    logger.info("Starting embedding and storing the chunks")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=DB_PATH)
    logger.info("Embedding and storing completed")
    return vector_db


def load_vectordb(embedding):
    logger.info("Loading existing database")
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding)


def get_vectordb(embedding):
    try:
        vector_db = None
        if os.path.exists(DB_PATH):
            vector_db = load_vectordb(embedding)
        else:
            chunks = load_document()
            vector_db = create_vectordb(chunks, embedding)
        return vector_db
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
