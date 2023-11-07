from typing import List
import pandas as pd
import numpy as np


def train_val_test_split(df, params, use_val: bool = False) -> List:
    """
    Split the dataframe into train, (optional validation), and test sets.

    :param df: DataFrame to split.
    :param params: Dictionary with split ratios under the key "split".
    :param use_val: Flag to determine if a validation set should be created.
    :return: A list of DataFrames containing the train, (validation), and test sets.
    """
    ratio = params.split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if use_val:
        train_cutoff = int(ratio[0] * len(df))
        val_cutoff = int((ratio[0] + ratio[1]) * len(df))
        train, val, test = np.split(df_shuffled, [train_cutoff, val_cutoff])
        return [train, val, test]

    test_cutoff = int((1 - ratio[2]) * len(df))
    train, test = np.split(df_shuffled, [test_cutoff])
    return [train, test]
