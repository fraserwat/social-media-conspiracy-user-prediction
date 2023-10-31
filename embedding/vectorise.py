from typing import List
from collections import Counter


def create_vectorised_layer(words: List, max_features: int):
    """Creates vectorisation function from the word list vocab"""
    # Step 1: Tokenize the words
    tokenized_words = [word.split() for word in words]
    flat_token_list = [token for sublist in tokenized_words for token in sublist]

    # Step 2: Build vocabulary and limit its size
    word_count = Counter(flat_token_list)
    vocab = {
        word: i + 2 for i, (word, _) in enumerate(word_count.most_common(max_features))
    }

    # Step 2.1: Out Of Vocab words assigned a 1
    oov_index = 1

    # Step 3: Create vectorize function
    def vectorize_func(text):
        return [vocab.get(token, oov_index) for token in text.split()]

    return vectorize_func
