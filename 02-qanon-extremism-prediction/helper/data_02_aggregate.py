from typing import List
import pandas as pd


def load_csvs(post_csvs: List[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(f"02-qanon-extremism-prediction/data/{csv}") for csv in post_csvs],
        ignore_index=True,
    )


def aggregate_non_bert_data() -> pd.DataFrame:
    """
    For the "task-specific word embeddings", we are just using the existing corpus, with no LLM or
    post-level sequencing.
    """
    ## Step 1: Load CSVs
    all_posts = load_csvs(["non-q-posts-v2.csv", "q-posts-v2.csv"])

    ## Step 2: Create New Column "words"
    def do_join(xs):
        return " ".join([s for s in xs if isinstance(s, str)])

    # all_posts["words"] = (
    #     all_posts["title"].astype(str) + " " + all_posts["selftext"].astype(str)
    # ).apply(do_join)
    all_posts["words"] = (
        all_posts["title"]
        .astype(str)
        .str.cat(all_posts["selftext"].astype(str), sep=" ")
    )

    all_posts.drop(["title", "selftext"], axis=1, inplace=True)
    all_posts.dropna(subset=["words", "q_level"], inplace=True)

    ## Step 3: Aggregate posts by author, with `words` being a sequence of all posts.
    grouped_posts = all_posts.groupby("hashed_author").agg(
        {"words": list, "q_level": "mean"}
    )

    return grouped_posts


def aggregate_bert_data(lstm=True) -> pd.DataFrame:
    """
    The standard embedding done on the above data is trained only on the dataset rather than making
    use of large pre-trained language models. They also do not leverage the fact that words are
    grouped into posts.

    This function prepares the Reddit data for use in sequential models like RNN and LSTM
    to make use of the post-level structure. For MLP models which can't do this, we just miss out
    the post-level aggregation
    """

    all_posts = load_csvs(["bert.csv"])
    grouped_posts = all_posts.copy()

    if lstm:
        grouped_posts = grouped_posts.groupby("post_id").agg(
            {"text": list, "author": "first", "q_level": "first"}
        )
    grouped_posts = grouped_posts.groupby("author").agg(
        {"text": list, "q_level": "first"}
    )

    return grouped_posts
