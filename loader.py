import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

files = ["ShahnwazHusain_Resume.pdf"]


def load_documents():
    chunks = []
    for file in files:
        try:
            # Load file
            loader = PyPDFLoader(file)
            document = loader.load()
            # Create Chunks from TEXT
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                      length_function=len)
            chunks.extend(splitter.split_documents(document))

        except FileNotFoundError:
            logger.error(f"File {file} not found in the root directory")
            raise
    return chunks
