import itertools
import numpy as np
from sklearn.model_selection import KFold
from training import model_init, model_tensors


def perform(static_params, grid_search_params, pos_path, neg_path, bert_path, cv=5):
    """
    Perform a grid search to find the best parameter configuration based on the F1 score.

    :param static_params: Unchanging configuration parameters for the model (i.e. outside gs).
    :param grid_search_params: Dictionary with parameter names as keys and lists of values to try.
    :param pos_path: File path to the positive samples dataset.
    :param neg_path: File path to the negative samples dataset.
    :param bert_path: File path to the BERT features dataset.
    :param cv: Number of cross-validation folds.

    :return: The best parameter configuration and its corresponding results.
    """

    param_names = grid_search_params.keys()
    param_combinations = list(itertools.product(*grid_search_params.values()))

    best_f1_score = -1
    best_combi_dict = None
    best_results = None
    loss, accuracy, recall, f1 = 0, 0, 0, 0
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Load training data for cross-validation
    data_dict = model_tensors.input_fn(
        pos_path=pos_path, neg_path=neg_path, bert_path=bert_path, params=static_params
    )
    words, labels = data_dict["train"]

    for idx, combi in enumerate(param_combinations):
        print(f"Grid search iteration {idx+1} of {len(param_combinations)}...")
        # Updating the non-grid params with those of the current grid search
        combi_dict = dict(zip(param_names, combi))
        current_params = static_params.copy()
        current_params.update(combi_dict)

        # cv scores, for each fold - later this will be averaged
        f1_scores = []
        k_num = 1

        for train_index, val_index in kfold.split(words):
            print(f"K-fold {k_num} of {cv}...")
            # Split data into k-folds for training and validation
            words_train, words_val = [words[i] for i in train_index], [
                words[i] for i in val_index
            ]
            labels_train, labels_val = labels[train_index], labels[val_index]

            # Update the inputs for the model function
            current_inputs = {
                "train": [words_train, labels_train],
                "val": [words_val, labels_val],
                "test": data_dict["test"],  # Test set remains constant across k-folds
            }

            # Train and evaluate the model
            loss, accuracy, recall, f1 = model_init.model_fn(
                current_inputs, current_params
            )
            f1_scores.append(f1)
            k_num += 1

        # Average F1 score across folds
        avg_f1_score = np.mean(f1_scores)
        print(f"Average F1 score: {avg_f1_score.round(4)}")

        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_combi_dict = combi_dict
            best_results = {
                "loss": loss,
                "accuracy": accuracy,
                "recall": recall,
                "f1": avg_f1_score,
            }

    return best_combi_dict, best_results
