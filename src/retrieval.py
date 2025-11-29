from typing import Optional
from qdrant_client import models

from src.config import COLLECTION_NAME
from src.client import qdrant_client, dense_model, sparse_model


def db_search(query: str, filter_condition: Optional[models.Filter] = None, limit: int = 5):
    dense_vectors = dense_model.embed_query(query)
    sparse_vectors = next(sparse_model.embed([query]))

    prefetch = [
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="sparse",
            limit=10,
        ),
        models.Prefetch(
            query=dense_vectors,
            using="dense",
            limit=10,
        )
    ]

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        query_filter=filter_condition
    )

    return response

