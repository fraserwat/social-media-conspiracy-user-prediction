import torch
from sentence_transformers import SentenceTransformer


def initialise(model_name: str = "all-MiniLM-L6-v2"):
    """Returns tokeniser and model objects required for all SBERT models"""
    tokeniser = None
    model = SentenceTransformer(model_name)
    model.eval()

    return tokeniser, model


def get_sbert_embeddings(model: SentenceTransformer, text: list):
    """Helper for encoding text with Sentence Transformer model"""
    return torch.tensor(model.encode(text, convert_to_tensor=True))
