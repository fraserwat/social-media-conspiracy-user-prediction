import pandas as pd
from typing import List


def load_csvs(post_csvs: List[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(f"../data/{csv}") for csv in post_csvs], ignore_index=True
    )


def filter_hashed_author_on_q_level(df: pd.DataFrame) -> pd.DataFrame:
    authors_with_q1 = df[df["q_level"] == 1]["hashed_author"].unique()

    # Filter Data: Keep rows with 'hashed_author's having at least one q_level=1
    df = df[df["hashed_author"].isin(authors_with_q1)]

    # Delete rows: For each 'hashed_author', keep rows before the first occurrence of q_level=1
    def filter_rows_before_q1(group):
        q1_index = group.index[group["q_level"] == 1].min()
        return group.loc[group.index < q1_index]

    return (
        df.groupby("hashed_author").apply(filter_rows_before_q1).reset_index(drop=True)
    )


def update_qlevel_response(df: pd.DataFrame) -> pd.DataFrame:
    def update_q_level(group):
        if group["q_level"].eq(1).any():
            group["q_level"] = 1
        return group

    return df.groupby("hashed_author").apply(update_q_level)


## Step 1: Load CSVs
all_posts = load_csvs(["non-q-posts-v2.csv", "q-posts-v2.csv"])

## Step 2: Create New Column "words"
all_posts["words"] = all_posts["title"] + " " + all_posts["selftext"]

## Step 3: Filter and Modify Data Based on "hashed_author" and "q_level"
all_posts = filter_hashed_author_on_q_level(df=all_posts)

## Step 4: Update
all_posts = update_qlevel_response(df=all_posts)
