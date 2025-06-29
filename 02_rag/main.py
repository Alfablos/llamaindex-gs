import sys
from os.path import isdir
from typing import Any, Generator

import chromadb
import fitz # pymupdf
# import pdfplumber
# import pypdf
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.core.readers.base import BaseReader
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# Make sure you have OPENAI_API_KEY in env
# Settings.llm = Ollama(
#     model='deepseek-r1:14b',
#     temperature=0.1,
#     request_timeout=150.0,
#     base_url='http://localhost:11434'
# ) # OpenAI(model="gpt-4.1", temperature=0)
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
# Local embeddings, remote LLM
Settings.embed_model = OllamaEmbedding(
    model_name='nomic-embed-text:v1.5',
    base_url='http://localhost:11434'
)


index = len(sys.argv) >= 2 and sys.argv[1] == 'index'

persistence_directory = './persistence'
data_directory = './docs'
db_path = persistence_directory + '/chroma_db'
db_collection = 'aws_docs'

# def pdf_loader(path) -> list[Document]:
#     docs = []
#     with pymupdf.open(path) as f:
#         for i, page in enumerate(f):
#             text = page.get_text()
#             if page:
#                 docs.append(Document(text=text, metadata={'page': i + 1}))
#     return docs
#
#
# def pdfplumber_loader(file_path):
#     docs = []
#     with pdfplumber.open(file_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 docs.append(Document(text=text, metadata={"page": i+1}))
#     return docs
#

# class PDFReader(BaseReader):
#     def lazy_load_data(self, file_path: Any, **load_kwargs: Any) -> Iterable[Document]:
#         docs = []
#         with pdfplumber.open(file_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 text = page.extract_text()
#                 if text:
#                     docs.append(Document(text=text, metadata={"page": i + 1}))
#         return docs

class PDFReader(BaseReader):
    def lazy_load_data(self, file_path: Any, **load_kwargs: Any) -> Generator[Document]:
        with fitz.open(file_path) as pdf:
            for i in range(pdf.page_count):
                yield Document(
                    text = pdf.load_page(i).get_text(),
                    metadata = { 'page': i + 1 }
                )


extractors_override = {
    '.pdf': PDFReader(),
}




if isdir(persistence_directory) and not index:
    print('Using existing vector store')
    # read db collection
    db = chromadb.PersistentClient(db_path)
    db_collection = db.get_or_create_collection(db_collection)
    # Initialize store
    vector_store = ChromaVectorStore(chroma_collection=db_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
else:
    print('Creating new vector store')

    # Create collection
    db = chromadb.PersistentClient(db_path)
    db_collection = db.get_or_create_collection(db_collection)
    # Initialize store
    vector_store = ChromaVectorStore(chroma_collection=db_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = SimpleDirectoryReader(
        input_dir=data_directory,
        recursive=True,
        file_extractor=extractors_override,
        filename_as_id=True
    )

    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

query_engine = index.as_query_engine()


def main():
    response = query_engine.query('what\'s the content of the provided documents?')
    print(str(response))


if __name__ == "__main__":
    main()
