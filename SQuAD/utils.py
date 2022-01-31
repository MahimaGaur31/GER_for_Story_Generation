# external libraries
import os
import re
import pickle
import string
import numpy as np
from collections import Counter
from spacy.lang.en import English
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# internal utilities
import config

tokenizer = English()
device = torch.device("cuda" if config.cuda else "cpu")


def clean_text(text):
    text = text.replace("]", " ] ")
    text = text.replace("[", " [ ")
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text


def word_tokenize(sent):
    return [token.text for token in tokenizer(sent)]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def build_vocab(context_filename, question_filename, word_vocab_filename, word2idx_filename,
                char_vocab_filename, char2idx_filename, is_train=True, max_words=-1):
    # select the directory we want to create the vocabulary from
    directory = config.train_dir if is_train else config.dev_dir

    # load the context and question files
    with open(os.path.join(directory, context_filename), "r", encoding="utf-8") as context,\
         open(os.path.join(directory, question_filename), "r", encoding="utf-8") as question:
        context_file = context.readlines()
        question_file = question.readlines()

    # clean and tokenize the texts
    words = [w.strip("\n") for doc in context_file + question_file for w in word_tokenize(clean_text(doc))]
    chars = [c for w in words for c in list(w)]
    # create a dictionary with word and char frequencies
    word_vocab = Counter(words)
    char_vocab = Counter(chars)
    # put them in a list ordered by frequency
    word_vocab = ["--NULL--"] + ["--UNK--"] + sorted(word_vocab, key=word_vocab.get, reverse=True)
    char_vocab = ["--NULL--"] + ["--UNK--"] + sorted(char_vocab, key=char_vocab.get, reverse=True)
    # limit the word vocabulary to top max_words
    word_vocab = word_vocab[:max_words]
    # get the word and char to ID dictionary mapping
    word2idx = dict([(x, y) for (y, x) in enumerate(word_vocab)])
    char2idx = dict([(x, y) for (y, x) in enumerate(char_vocab)])

    # save those files
    with open(os.path.join(directory, word_vocab_filename), "wb") as wv, \
         open(os.path.join(directory, word2idx_filename), "wb") as wd, \
        open(os.path.join(directory, char_vocab_filename), "wb") as cv, \
        open(os.path.join(directory, char2idx_filename), "wb") as cd:
        pickle.dump(word_vocab, wv)
        pickle.dump(word2idx, wd)
        pickle.dump(char_vocab, cv)
        pickle.dump(char2idx, cd)

    print("Vocabulary created successfully.")
    return word_vocab, word2idx, char_vocab, char2idx


def build_embeddings(vocab, embedding_path="", output_path="", vec_size=50):
    embedding_dict = {}
    # Load pretrained embeddings if an embedding path is provided
    if embedding_path:
        # Get the path associated to the embedding size we want
        embedding_path = embedding_path.format(vec_size)
        with open(embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                if word in vocab:
                    embedding_dict[word] = vector

    embedding_dict["--NULL--"] = np.asarray([0. for _ in range(vec_size)])
    embedding_dict["--UNK--"] = np.asarray([0. for _ in range(vec_size)])
    embedding_matrix = []
    count = 0
    for v in vocab:
        if v in embedding_dict:
            embedding_matrix.append(embedding_dict[v])
        else:
            count += 1
            embedding_matrix.append(np.random.normal(0, 0.1, vec_size))
    # Save the embedding matrix
    with open(os.path.join(config.train_dir, output_path), "wb") as e:
        pickle.dump(embedding_matrix, e)


