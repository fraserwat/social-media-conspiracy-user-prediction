# Using pre-Q sequences of reddit posts to predict user-level QAanon participation

**Original Paper**: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf

**Original Codebase:** https://github.com/isvezich/cs230-political-extremism

**Authors**: Lillian Ma, Stephanie Vezich

## What can I learn from this paper?

- [ ] Natural Language Processing
- [ ] Pros / Cons of NN Architectures (RNN vs LSTM).

## Files

- `reddit_data.py` - Having been given a list of download URLs, can just run as a helper function.

## Paper Structure

### Abstract

- Two sentences giving a quick overview of the report's context.
- What are we planning to do here? _("The goal of the current work is to investigate whether it is possible to predict extremist group membership from the language of this preceding participation")_.
- What was our methodology? _("Using the reddit Pushshift API, we collect all posts authored by a sample of users who either later joined QAnon subreddits or did not.")_
- What did we find? _("We then use language embeddings to predict class membership, achieving 79% accuracy and 0.80 F1-score")_.
- Two sentences on different architectures tried.
- What do these findings suggest? _("These findings suggest that the sequence of a user’s posts may encode their development of extremist attitudes, and provide a method for online community platforms to identify and potentially intervene on at-risk users.")_

### Introduction

- First paragraph here is purely contextual, with lots of sources.
- Second paragraph brings in the platform (reddit) and links the problem context up with a potential data science use case _("If it is possible to identify a priori who might develop extremist beliefs... one might imagine modifying recommendation algorithms accordingly to steer at-risk users toward more moderate groups")_
- How are you going to test your hypothesis? _("To test this hypothesis, we are investigating whether we can predict the probability of subsequent QAnon membership from pre-Q reddit posts in other (non QAnon related) subreddits.")_

### Related Work

- First paragraph compares and contrasts other papers which can be drawn on.
- Where are there gaps in the current literature?
- Are there any criticisms of the current methodology which have been made by experts, but not covered by new research?

### Dataset and Features

- This is a binary classification task, where the positive class is QAnon engagement. In order to create this dataset, you need both Q and NonQ users.
- It mentions using SQL to join a Q-specific dataset with a generalised Reddit dataset as not to bias in favour of Q posts, but this seems to be done before the creation of `q-posts-v2.csv` and `non-q-posts-v2.csv`?
- "Pushshift" is a public dataset that contains all reddit posts since 2006, which was used for this. Future replication of this project would presumably not be as easy due to changes to Reddit's API policy?

### Method

- In terms of the _"Figure 1: Training Examples"_ image, this "words" column is a concatenation of the title and "selftext" (body of the post). This is defined in the `input_fn.py` file, in the `oad_data_to_df()` fn.
- **_Windowing:_** They were only interested in the posts which LED UP TO the first post in a QAnon subreddit, so any posts after the first Q-post were removed. This also removed any users (~50%) whose first post was Qanon related.
- **_Filtering:_** Weird bot behaviour identified and users deleted.
- **_Aggregation:_** Each post treated as a string (see above) in an array of all posts for a given user. Some basic summary statistics given for postitive and negative class labels.

#### Embedding

- Baseline models use text standardisation and then create vector of sequenced word embeddings.
- The above has two drawbacks:
  1. Only based on existing texts, further context can be gained with Large Language Models (LLMs).
  2. Does not leverage the fact that words are grouped into posts.
- Solves both using SBERT, treating each post as a sentence.

  - As SBERT has quadratic increasing memory and time consumption + posts can be ≤10k words, they did not encode entire posts.

  - Each post were taken as 150 word chunks, and then fed to the SBERT transformer. This list of 150 word chunks then returns a 384-dimensional embedding for each chunk.

  - A baseline takes an average, but the main models keep this information to preserve sequential information across chunks.

### Model Architectures

## Code Review

- Probably worth noting that the code provided is a lot more convoluted than the neat examples I've seen so far in online courses. So even working out what is doing what is a bit of a challenge.
- Everything seems to come together in `/train.py`, with the other files mostly being either configurations for different models (in a `model_name/params.json` format), or helper functions
