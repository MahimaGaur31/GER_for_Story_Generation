# Studying the Effect of Generalized Entropy Regularization on Story Generation

Term Project for CSCI-GA 2590: Natural Language Processing @ NYU

This project aims to study the effectiveness of using [Generalised Entropy Regularizers](https://arxiv.org/pdf/2005.00820.pdf) on Story Generation.

# Setup

## Read and Preprocess Data (SQuAD)

Reference: [GauthierDmn's Code Implementation](https://github.com/GauthierDmn/question_answering)

1. Modify `config.py` to specify data directory paths
2. Run `python -m spacy download en`
3. Run `python make_dataset.py`

`make_dataset.py` reads in the SQuAD v2.0 data and seperates the fields into answers, questions, labels and contexts. Contexts refers to the Wikipedia articles used to derive questions and snwers, while labels returns the index of the tokens in the answers.


# Useful Links
- [GER paper](https://arxiv.org/pdf/2005.00820.pdf)
- [Story Generation Paper](https://arxiv.org/pdf/1805.04833.pdf)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [GauthierDmn's Code Implementation](https://github.com/GauthierDmn/question_answering)
