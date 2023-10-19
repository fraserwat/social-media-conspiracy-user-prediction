import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import data_02_aggregate as dagg


sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # .to("cuda")
sbert_model.eval()  # set to evaluation mode


def get_sbert_embeddings(text: list):
    return torch.tensor(sbert_model.encode(text, convert_to_tensor=True))


def flatten_and_filter_sentences(nested_sentences, sentence_length):
    flat_sentences = [
        sentence
        for sublist in nested_sentences
        for sentence in sublist
        if isinstance(sentence, str) and len(sentence.strip()) > 0
    ]
    return flat_sentences[:sentence_length]


def remaining_time(start_time, idx, n_idx):
    estimated_total_time = ((time.time() - start_time) / (idx + 1)) * n_idx
    time_remaining = estimated_total_time - (time.time() - start_time)
    # Convert estimated remaining time from seconds to HH:MM:SS format
    hours, remainder = divmod(time_remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def convert_bert_rnn_mlp_df_to_tensor(df, params):
    authors = []
    sentences_length = params.get("sentences_length", 150)

    print("Performing rnn_mlp SBERT Embedding...")
    start_time = time.time()
    for idx, nested_sentences in enumerate(df["text"]):
        valid_sentences = flatten_and_filter_sentences(
            nested_sentences, sentences_length
        )
        if not valid_sentences:
            continue
        sentences_embedding = get_sbert_embeddings(valid_sentences)
        if idx % 2**5 == 0:
            torch.cuda.empty_cache()

        print(
            f"Embedded row {idx + 1} of {len(df['text'])}. Est: {remaining_time(start_time, idx, len(df['text']))} ({round(100 * (idx + 1) / len(df['text']), 1)}%)"
        )
        authors.append(sentences_embedding)

    print("...Completed SBERT Embedding!")

    # Find the global max number of sentences to pad all authors to the same shape.
    max_sentences = max(embedding.shape[0] for embedding in authors)

    padded_authors = [
        torch.cat(
            [
                author_embedding,
                torch.zeros(
                    max_sentences - author_embedding.shape[0], author_embedding.shape[1]
                ),
            ],
            dim=0,
        )
        for author_embedding in authors
    ]

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
    embedding_dim = authors[0][0].shape[1]

    for author in authors:
        padded_author = []

        # Inner padding: Ensure all posts have same number of chunks.
        for post in author:
            padding = torch.zeros(max_chunks - post.shape[0], embedding_dim)
            padded_post = torch.cat([post, padding], dim=0)
            padded_author.append(padded_post)

        # Outer padding: Ensure all authors have same number of posts.
        while len(padded_author) < max_posts:
            # Pad using zeros. Ensure it has the correct shape [max_chunks, embedding_dim].
            padded_author.append(torch.zeros(max_chunks, embedding_dim))

        # Combine padded posts into a tensor and add it to the padded_authors list.
        padded_authors.append(torch.stack(padded_author))

    # Combine all authors into a single tensor.
    return torch.stack(padded_authors)


def convert_to_tensor(df: pd.DataFrame):
    # need to flattern into a list of words
    df["tokens"] = df["words"].apply(
        lambda posts: [word for post in posts for word in post.split()]
    )
    all_tokens = [
        token for tokens_list in df["tokens"].tolist() for token in tokens_list
    ]
    vocab = set(all_tokens)

    # Map our vocab to integer idenfiers
    token_to_id = {token: i for i, token in enumerate(vocab)}
    # Embedding layers don't take str, to get embeddings for each word need to convert text into IDs using vocab mapping
    df["token_ids"] = df["tokens"].apply(
        lambda tokens: [token_to_id[token] for token in tokens]
    )

    dataloader = DataLoader(
        list(df["token_ids"]), batch_size=32, shuffle=False, collate_fn=lambda x: x
    )

    # 3. Embedding & Averaging
    embeddings = torch.nn.Embedding(len(vocab), 512)

    user_embeddings = []
    for batch in dataloader:
        batch_embeddings = [embeddings(torch.tensor(user)).mean(0) for user in batch]
        user_embeddings.extend(batch_embeddings)

    # Convert q_levels to a tensor
    q_levels_tensor = (
        torch.tensor(df["q_level"].values).float().unsqueeze(1)
    )  # unsqueeze to make it a column vector

    # Concatenate the user_embeddings_tensor and q_levels_tensor along columns
    final_tensor = torch.cat((torch.stack(user_embeddings), q_levels_tensor), dim=1)

    return final_tensor


def embeddings(params: dict) -> torch.Tensor:
    # Fetching the right pre-processed dataset based on a BERT or non-BERT embedding.
    print("Loading QAnon dataset and creating df...")
    if params["model"].startswith("BERT"):
        print("Making bert dataset")
        df = dagg.aggregate_bert_data()
    else:
        print("Making word embedding dataset")
        df = dagg.aggregate_non_bert_data()

    print(df.head())

    # @TODO: Get rid of this line, get the tensor models into Google Drive or something.
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_shuffled.iloc[:2500]

    # Converting dataset to numerical tensor.
    if params["model"] == "BERT_LSTM":
        words = convert_bert_lstm_df_to_tensor(df, params=params)
    elif params["model"].startswith("BERT"):
        # embedding for BERT_RNN and BERT_MLP is identical.
        words = convert_bert_rnn_mlp_df_to_tensor(df, params=params)
    else:
        # This will just be MLP
        words = convert_to_tensor(df=df)

    return words


# for mod in ["TASK_SPECIFIC", "BERT_LSTM", "BERT_RNN"]:
for mod in ["MLP"]:
    torch.save(
        embeddings(params={"model": mod}),
        f"02-qanon-extremism-prediction/data/{mod}.pth",
    )


# Load the tensor or model from the .pth file
tensor_or_model = torch.load("02-qanon-extremism-prediction/data/MLP.pth")
# if isinstance(tensor_or_model, torch.Tensor):
print(tensor_or_model)
print(tensor_or_model.shape)

# tensor_or_model = torch.load("02-qanon-extremism-prediction/data/BERT_RNN.pth")
# if isinstance(tensor_or_model, torch.Tensor):
#     print(tensor_or_model.shape)
