import pandas as pd


def load_reddit_df(path):
    """Loading source data and handling data to group by author"""
    df = pd.read_csv(path, compression="gzip")

    # convert types (originally all strings) & filter features to date range before first q drop
    df = df.groupby("hashed_author").agg(
        {
            "title": list,
            "selftext": list,
            "q_level": "mean",
        }
    )

    # words is title + " " + selftext
    df["words"] = df["title"] + df["selftext"]

    # concatenate words into 1 string to create embedding from ALL words by author
    df["words"] = df["words"].apply(
        lambda x: " ".join([s for s in x if isinstance(s, str)])
    )

    # Get rid of component cols pre-words and any null words / q_levels
    df.drop(["title", "selftext"], axis=1)
    df.dropna(subset=["words", "q_level"], inplace=True)

    return df


def load_bert_data_to_df(path, params):
    df = pd.read_csv(path, compression="gzip")

    # Extra sequencing required if LSTM
    if params.get("model", "BERT_MLP") == "BERT_LSTM":
        print("BERT_LSTM: Grouping by post_id")
        df = df.groupby("post_id").agg(
            {"text": list, "author": "first", "q_level": "first"}
        )
        df = df.groupby("author").agg({"text": list, "q_level": "first"})
        df = df.sample(frac=params.get("sample_rate", "1.0")).reset_index()
    else:
        df = df.groupby("author").agg({"text": list, "q_level": "first"})
        df = df.sample(frac=params.get("sample_rate", "1.0")).reset_index()
    print(f"length of features: {len(df)}")

    return df


def load_all_data_task_specific(pos_path, neg_path, params):
    """Top level handler for non-BERT embedding cases"""
    pos = load_reddit_df(pos_path)
    neg = load_reddit_df(neg_path)
    features = pd.concat([pos, neg], axis=0)
    features = features.sample(frac=params["sample_rate"]).reset_index()
    print(f"length of features: {len(features)}")

    return features
