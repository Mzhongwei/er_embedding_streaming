import multiprocessing as mp

import gensim.models as models
import numpy as np
from gensim.models import Doc2Vec, FastText, Word2Vec


def initialize_embeddings(
    write_walks,
    dimensions,
    window_size,
    training_algorithm="word2vec",
    learning_method="skipgram",
    workers=mp.cpu_count(),
    sampling_factor=0.001,
):
    """Function used to train the embeddings based on the given walks corpus. Multiple parameters are available to
    tweak the training procedure. The resulting embedding file will be saved in the given path to be used later in the
    experimental phase.

    :param output_embeddings_file: path to save the embeddings file into.
    :param walks: path to the walks file (if write_walks == True), list of walks otherwise.
    :param write_walks: flag used to read walks from a file rather than taking them from memory.
    :param dimensions: number of dimensions to be used when training the model
    :param window_size: size of the context window
    :param training_algorithm: either fasttext or word2vec.
    :param learning_method: skipgram or CBOW
    :param workers: number of CPU workers to be used in during the training. Default = mp.cpu_count().
    """
    if training_algorithm == "word2vec":
        if learning_method == "skipgram":
            sg = 1
        elif learning_method == "CBOW":
            sg = 0
        else:
            raise ValueError("Unknown learning method {}".format(learning_method))
        if write_walks:
            model = Word2Vec(
                vector_size=dimensions,
                window=window_size,
                min_count=2,
                sg=sg,
                workers=workers,
                sample=sampling_factor,
            )
            return model
        else:
            model = Word2Vec(
                vector_size=dimensions,
                window=window_size,
                min_count=2,
                sg=sg,
                workers=workers,
                sample=sampling_factor,
            )
            return model
    elif training_algorithm == "doc2vec":
        if learning_method == "skipgram":
            sg = 1
        elif learning_method == "CBOW":
            sg = 0
        else:
            raise ValueError("Unknown learning method {}".format(learning_method))
        if write_walks:
            model = Doc2Vec(
                size=dimensions,
                window=window_size,
                min_count=2,
                sg=sg,
                workers=workers,
                sample=sampling_factor,
            )
            return model
        else:
            model = Doc2Vec(
                size=dimensions,
                window=window_size,
                min_count=2,
                sg=sg,
                workers=workers,
                sample=sampling_factor,
            )
            return model
    elif training_algorithm == "fasttext":
        print("Using Fasttext")
        if write_walks:
            model = FastText(
                window=window_size,
                min_count=2,
                workers=workers,
                vector_size=dimensions,
            )
            return model
        else:
            model = FastText(
                vector_size=dimensions,
                workers=workers,
                min_count=2,
                window=window_size,
            )
            return model
