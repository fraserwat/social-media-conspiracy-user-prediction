from typing import List
import pandas as pd
import numpy as np


def train_val_test_split(df: pd.DataFrame, params: dict) -> List:
    ratio = params.get("split", [0.7, 0.15, 0.15])
    train_cutoff, val_cutoff = ratio[0], ratio[0] + ratio[1]
    splits = np.split(
        df.sample(frac=1), [int(train_cutoff * len(df)), int(val_cutoff * len(df))]
    )

    return splits
