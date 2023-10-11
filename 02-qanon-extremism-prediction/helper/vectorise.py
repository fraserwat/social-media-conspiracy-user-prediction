import typing
from sentence_transformers import SentenceTransformer

# import torch
# from sklearn.metrics.pairwise import cosine_similarity


def chunk_users_posts(post: str, chunk_size: int = 150) -> typing.List[str]:
    """
    Helper function for keeping SBERT performant, and transformer input array relatively small.
    """
    post_words = post.split(" ")
    chunks = [
        " ".join(post_words[i : i + chunk_size])
        for i in range(0, len(post_words), chunk_size)
    ]
    return chunks


def generate_sbert_embeddings(post: str, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    chunks = chunk_users_posts(post)
    return model.encode(chunks)
