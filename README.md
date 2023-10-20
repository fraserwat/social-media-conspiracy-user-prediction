# Using Sequences of Reddit Posts to Predict User-Level QAnon Participation

This project replicates the findings in a paper predicting a user's participation in conspiracy theory subreddits using past posts. I have expanded on the original paper in three substantial ways:

1. Due to the [growing dominance of PyTorch](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/) I have refactored everything from Tensorflow to PyTorch. As Tensorflow tends to be a more high-level and abstracted library, there were a lot of one-line TF functions or class methods I had to write myself.
1. The LLM-boom of the last year largely happened after the publishing of this paper. I have experimented with other SBERT transformers in an attempt to improve on the original accuracy score.
1. In an effort to improve model interpretability, I have implemented explanatory outputs using Retrieval Augmented Generation (RAG). Our primary model gives a prediction, then our RAG tells us why.

**Original Paper**: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf

**Paper GitHub Repo:** https://github.com/isvezich/cs230-political-extremism

## Key Skills Developed from Project

- Natural Language Processing.
- Building Machine Learning Pipelines on DataBricks.
- Understanding of why we use different Deep Learning architectures (e.g. RNN vs LSTM).
- Retrieval Augmented Generation.

#### @TODO: EVERYTHING UNDER THIS SHOULD BE REWRITTEN TO SHOW PEOPLE HOW TO REPLICATE / QUICK REVIEW OF PERFORMANCE / CHANGES.

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

- Everything seems to come together in `/train.py`, with the other files mostly being either configurations for different models (in a `model_name/params.json` format), or helper functions
- Data is loaded in `model/input_fn.py`, and it is unclear how much of the processing discussed in the paper is done before the csvs are created, and how much is done in the code base. Process seems to go as follows:

  1. `input_fn` takes in the positive and negative case dataframes, the BERT dataframe, and any model parameters.
  1. If it is not a BERT model, the positive and negative case dataframes are fed to `load_all_data_to_df`.
  1. Within `load_all_data_to_df`, `pos` and `neg` variables declared as the response to feedint the positive and negative case dataframes through the `load_data_to_df` function.
  1. Judging by the SQL queries [sample_non_q_author_posts.sql](https://github.com/isvezich/cs230-political-extremism/blob/c8950ad69023a9e3e52f25b520da84499d530cc8/data/db/sql/sample_non_q_author_posts.sql) and [sample_q_author_posts.sql](https://github.com/isvezich/cs230-political-extremism/blob/c8950ad69023a9e3e52f25b520da84499d530cc8/data/db/sql/sample_q_author_posts.sql) it seems like the filtering on start and pre-q date has already been done for us.

* Did some of my own filtering, but in the end the only way to get the q_level 0/1 split they had was to assume all the data processing had already been done.
* There was an issue with the original code I assume is a mistake given the context from the paper - in the below code block, all of the titles and post texts (`selftext`) are aggregated _and then_ concatonated. This means that sequentially, all of a users post titles occur before the post content.

```{python}
def load_data_to_df(path):
    df = pd.read_csv(path, compression='gzip')
    # convert types (originally all strings) & filter features to date range before first q drop
    df = df \
        .groupby("hashed_author") \
        .agg(
        {
            "title": lambda x: list(x),
            "selftext": lambda x: list(x),
            "q_level": "mean",
        }
    )

    # concatenate title & body text into 1 string to create embedding from all the words author ever wrote
    def do_join(xs):
        return " ".join([s for s in xs if type(s) == str])

    df["words"] = (df["title"] + df["selftext"]).apply(do_join)
    df.dropna(subset=['words', 'q_level'], inplace=True)

    return df

```

I have switched around the aggregation vs creation of `df["words"]` so that the sequential order for a given author is title1, text1, title2, text2, etc. This doesn't matter for the MLP model, but sequencing does matter for RNN and LSTM models.

For the `bert.csv`, the number of rows in this CSV is ~1% out of N `non-q-posts-v2.csv` + N `q-posts-v2.csv`, so can assume that the required preprocessing has already been done for this (also as the non-q and q posts don't have a post id in the given CSVs, would run the risk of grouping different posts with the same title together).
