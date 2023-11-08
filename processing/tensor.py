def convert_bert_rnn_mlp_to_tensor(df, params):
    authors = []
    max_length = params.sentences_length

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
    authors_data = []
    max_posts = params.posts_length
    max_length = params.sentences_length

    # Iterate over each author's list of posts
    for _, author_posts in df["text"].items():
        # Process each post, which is a list of comments
        processed_posts = []
        for post_comments in author_posts[:max_posts]:
            # Concatenate all comments in a post into a single string, truncate, and lowercase
            filtered_comments = [
                str(comment) for comment in post_comments if isinstance(comment, str)
            ]
            concatenated_comments = " ".join(filtered_comments).lower()
            truncated_comments = " ".join(concatenated_comments.split()[:max_length])
            processed_posts.append(truncated_comments)

        # Add the processed posts for this author to the main list
        authors_data.append(processed_posts)

    # Return the structured data for use with a DataLoader
    return authors_data
