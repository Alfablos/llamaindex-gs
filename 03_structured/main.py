from typing import List
from pathlib import Path
import json
from pydantic import BaseModel, Field
from datetime import datetime
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


llm = Ollama(
    model='qwen3:14b',
    temperature=0.1,
    request_timeout=150.0,
    base_url='http://localhost:11434'
) # OpenAI(model="gpt-4.1", temperature=0)

# Local embeddings, remote LLM
Settings.embed_model = OllamaEmbedding(
    model_name='nomic-embed-text:v1.5',
    base_url='http://localhost:11434'
)

pdf_reader = PyMuPDFReader()

class Ticket(BaseModel):
    nome_passeggero: str = Field(
        description='Il nome del passeggero'
    )
    stazione_partenza: str = Field(
        description='La stazione da cui parte il treno'
    )
    stazione_arrivo: str = Field(
        description='La stazione di arrivo'
    )
    data_partenza: str = Field(
        description='Data della partenza, ad esempio "10 OTT 2025" per indicare 10 Ottobre 2025'
    )
    carrozza: int = Field(
        description='Il numero della carrozza in cui viaggia il passeggero'
    )
    numero_posto: int = Field(
        description='Il numero del posto assegnato al passeggero'
    )
    prezzo: float = Field(
        description='Il prezzo del viaggio. Accanto alla cifra di solito si trova la valuta espressa in simboli o sigle, per esempio $/USD o €/EUR'
    )

sllm = llm.as_structured_llm(Ticket)

documents = pdf_reader.load_data(file_path=Path('./tickets/ticket1.pdf'), metadata=True)




def main():
    response = sllm.complete(documents[0].text)
    json_resp = json.loads(response.text)
    print(json.dumps(json_resp, indent=2))


if __name__ == "__main__":
    main()
