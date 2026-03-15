import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, FILE_NAME

logger = logging.getLogger(__name__)


def load_document():
    try:
        # Load file
        loader = PyPDFLoader(FILE_NAME)
        document = loader.load()
        # Create Chunks from TEXT
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                  length_function=len)
        chunks = splitter.split_documents(document)
        return chunks
    except FileNotFoundError:
        logger.error(f"File {FILE_NAME} not found in the root directory")
        raise
