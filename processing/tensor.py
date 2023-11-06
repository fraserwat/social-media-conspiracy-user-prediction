def convert_bert_rnn_mlp_to_tensor(df, params):
    authors = []
    max_length = params.get("sentences_length", 150)

    # for i, sentences in enumerate(df["text"]):
    for author_posts in df["text"]:
        # Truncate each post individually to max_length words and filter out non-string types
        sentences = [post for post in author_posts if isinstance(post, str)]
        truncated_sentences = [
            " ".join([w.lower() for w in post.split()[:max_length]])
            for post in sentences
        ]

        # Original uses ragged tensors. No equivalent exists in PyTorch, so will pad later.
        authors.append(truncated_sentences)

    # In PyTorch, can just leave sequences as lists and use directly in data loaders
    return authors


def convert_bert_lstm_to_tensor(df, params):
    authors = []

    # 1. Iterates over each author's text items.
    for idx, au in enumerate(df["text"]):
        posts = []
        # 2. Within each author, posts are iterated over, truncated and cleaned.
        for jdx, post in enumerate(au):
            if jdx < params.get("posts_length", 50):
                # Truncate the post sentences to the specified length
                truncated_post = post[: params.get("sentences_length", 150)]

                # Convert to lowercase and filter out non-string types if needed
                lower_cased_post = [
                    w.lower() for w in truncated_post if isinstance(w, str)
                ]
                # 3. Each cleaned, truncated post is then appended to a list for that author.
                posts.append(lower_cased_post)

        # 4. Finally, all of the authors' posts are appended to an overall list.
        authors.append(posts)

    # In PyTorch, you can often leave the sequences as lists and use those directly in data loaders
    return authors
