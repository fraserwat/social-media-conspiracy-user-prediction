# Using Sequences of Reddit Posts to Predict User-Level QAnon Participation

This project replicates the findings in a paper predicting a user's participation in conspiracy theory subreddits using past posts. I have expanded on the original paper in three substantial ways:

1. Due to the [growing dominance of PyTorch](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/) I have refactored everything from Tensorflow to PyTorch. As Tensorflow tends to be a more high-level and abstracted library, there were a lot of one-line TF functions or class methods I had to write myself.
1. The LLM-boom of the last year largely happened after the publishing of this paper. I have experimented with other SBERT transformers in an attempt to improve on the original accuracy score.
1. In an effort to improve model interpretability, I have implemented explanatory outputs using Retrieval Augmented Generation (RAG). Our primary model gives a prediction, then our RAG tells us why.

**Original Paper**: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf

**Paper GitHub Repo:** https://github.com/isvezich/cs230-political-extremism

TODO: Note that when you use padded sequences with LSTM or other RNN layers, you might also want to use packing to make the computation more efficient and accurate, but that's an advanced topic.
