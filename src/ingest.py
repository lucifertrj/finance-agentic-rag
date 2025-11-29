import uuid
import gc

from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFium2Parser
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    Modifier,
    BinaryQuantization,
    BinaryQuantizationConfig,
    PointStruct,
)
from qdrant_client import models

from src.config import COLLECTION_NAME, DATA_PATH
from src.client import qdrant_client, dense_model
from src.utils import update_metadata


def load_documents(path: str = DATA_PATH):
    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=path,
            glob="**/*.pdf",
        ),
        blob_parser=PyPDFium2Parser(),
    )
    return loader.load()


def create_collection():
    check_dim = dense_model.embed_query("testing the dimensions of embedding model")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=len(check_dim),
                distance=Distance.COSINE,
                on_disk=True
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                modifier=Modifier.IDF,
            ),
        },
        quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(always_ram=False)
        )
    )


def create_payload_indexes():
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_type",
        field_schema=models.PayloadSchemaType.KEYWORD
    )


def index_documents(data):
    for i, doc in enumerate(data):
        try:
            tags = doc.metadata.get("chunk_tags", "")
            content = doc.page_content.strip()

            searchable_text = f"{content} \n Keywords: {tags}"

            dense_embedding = dense_model.embed_query(searchable_text)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embedding,
                    "sparse": models.Document(
                        text=searchable_text,
                        model="Qdrant/bm25",
                    )
                },
                payload={
                    "content": content,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get('page', 0),
                    "chunk_tags": tags,
                    "document_type": doc.metadata.get("doc_type", ""),
                    "chunk_id": doc.metadata.get('chunk_id', ''),
                    "calendar_year": doc.metadata.get('calendar_year', ''),
                }
            )

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point],
            )

            if i % 10 == 0:
                gc.collect()

        except Exception as e:
            print(e)


def run_ingestion(data_path: str = DATA_PATH):
    documents = load_documents(data_path)
    modified_data = update_metadata(documents)
    create_collection()
    index_documents(modified_data)
    create_payload_indexes()
    return len(modified_data)

