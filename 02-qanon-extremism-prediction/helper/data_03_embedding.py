import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import data_02_aggregate as dagg


sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sbert_embeddings(text: list):
    return torch.tensor(sbert_model.encode(text))


def flatten_and_filter_sentences(nested_sentences, sentence_length):
    flat_sentences = [
        sentence[:sentence_length]
        for sublist in nested_sentences
        for sentence in sublist
        if isinstance(sentence, str) and len(sentence.strip()) > 0
    ]
    return flat_sentences


def convert_bert_rnn_mlp_df_to_tensor(df, params):
    authors = []
    sentences_length = params.get("sentences_length", 150)

    print("Performing rnn_mlp SBERT Embedding...")
    for nested_sentences in df["text"]:
        valid_sentences = flatten_and_filter_sentences(
            nested_sentences, sentences_length
        )

        # Ensuring sentences are valid and non-empty
        if not valid_sentences:
            print(f"No valid sentences found for entry: {nested_sentences}")
            continue

        sentences_embedding = get_sbert_embeddings(valid_sentences)
        authors.append(sentences_embedding)

    print("...Completed SBERT Embedding!")

    # Find the global max number of sentences to pad all authors to the same shape.
    max_sentences = max(embedding.shape[0] for embedding in authors)

    padded_authors = []
    for author_embedding in authors:
        # Padding: Ensure all authors have the same number of sentences.
        padding = torch.zeros(
            max_sentences - author_embedding.shape[0], author_embedding.shape[1]
        )
        padded_author = torch.cat([author_embedding, padding], dim=0)

        padded_authors.append(padded_author)

    # Combine all authors into a single tensor.
    return torch.stack(padded_authors)


def convert_bert_lstm_df_to_tensor(df, params):
    authors = []
    max_posts_per_author = params.get("posts_length", 100)

    # Collect all the author_tensors first to find global max number of posts and chunks per post.
    print("Performing SBERT Embedding...")
    for _, posts in df["text"].items():
        author_tensor = []

        # Process up to `max_posts_per_author` posts.
        for post in posts[:max_posts_per_author]:
            valid_chunks = [chunk for chunk in post if isinstance(chunk, str)]
            assert len(valid_chunks) > 0 and all(
                isinstance(chunk, str) for chunk in valid_chunks
            ), f"Post contains no valid string values: {post}"

            post_embedding = get_sbert_embeddings(valid_chunks)
            author_tensor.append(post_embedding)

        authors.append(author_tensor)
    print("...Completed SBERT Embedding!")

    # Find the global max number of posts and chunks to pad all authors to the same shape.
    max_posts = max(len(author) for author in authors)
    max_chunks = max(tensor.shape[0] for author in authors for tensor in author)

    padded_authors = []
    for author in authors:
        padded_author = []

        # Inner padding: Ensure all posts have same number of chunks.
        for post in author:
            padding = torch.zeros(max_chunks - post.shape[0], post.shape[1])
            padded_post = torch.cat([post, padding], dim=0)
            padded_author.append(padded_post)

        # Outer padding: Ensure all authors have same number of posts.
        while len(padded_author) < max_posts:
            # Pad using zeros. Ensure it has the correct shape [max_chunks, embedding_dim].
            padded_author.append(torch.zeros(max_chunks, post.shape[1]))

        # Combine padded posts into a tensor and add it to the padded_authors list.
        padded_authors.append(torch.stack(padded_author))

    # Combine all authors into a single tensor.
    return torch.stack(padded_authors)


def convert_to_tensor(column):
    vectorizer = CountVectorizer().fit_transform(column.apply(" ".join))
    return torch.from_numpy(vectorizer.toarray())


def embeddings(params: dict) -> pd.DataFrame:
    # Fetching the right pre-processed dataset based on a BERT or non-BERT embedding.
    print("Loading QAnon dataset and creating df...")
    if params["model"].startswith("BERT"):
        print("Making bert dataset")
        df = dagg.aggregate_bert_data()
    else:
        print("Making word embedding dataset")
        df = dagg.aggregate_non_bert_data()

    # Converting dataset to numerical tensor.
    if params["model"].startswith("BERT"):
        if params["model"] == "BERT_LSTM":
            words = convert_bert_lstm_df_to_tensor(df, params=params)
        else:
            # At this stage in the processing, the embedding for BERT_RNN and BERT_MLP is the same.
            words = convert_bert_rnn_mlp_df_to_tensor(df, params=params)
    else:
        words = convert_to_tensor(df["words"])

    return words


print(embeddings(params={"model": "BERT_RNN", "posts_length": 1}).shape)
