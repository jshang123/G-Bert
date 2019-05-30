from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from build_tree import build_icd9_tree, build_atc_tree
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from predictive_models import TSNE

from sklearn.metrics import jaccard_similarity_score, f1_score, roc_auc_score, average_precision_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = '../saved/tsne'
# model_name = 'pretraining-LARGE-Re-Re'
model_name = '../saved/GBert-predict'


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.rx_voc = self.add_vocab(os.path.join(data_dir, 'rx-vocab.txt'))
        self.dx_voc = self.add_vocab(os.path.join(data_dir, 'dx-vocab.txt'))

        self.rx_singe2multi = []
        with open(os.path.join(data_dir, 'rx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rx_singe2multi.append(
                    self.rx_voc.word2idx[code.rstrip('\n')])

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


def save():
    tokenizer = EHRTokenizer(data_dir='../data')
    logger.info("Use Pretraining model")
    model = TSNE.from_pretrained(model_name, dx_voc=tokenizer.dx_voc,
                                 rx_voc=tokenizer.rx_voc)
    model(output_dir=output_dir)
    logger.info('# of model parameters: ' + str(get_n_params(model)))


def generate_meta(build_tree_func, task, output_path='emb-meta.tsv'):
    tokenizer = EHRTokenizer(data_dir='../data')
    voc = tokenizer.dx_voc if task == 0 else tokenizer.rx_voc
    res, graph_voc = build_tree_func(list(voc.idx2word.values()))

    # get level
    level_dict = {}
    for row in res:
        for level, item in enumerate(row):
            level_dict[item] = level

    with open(os.path.join(output_dir, ('dx-' if task == 0 else 'rx-') + output_path), 'w') as fout:
        fout.write('name\tlevel\n')
        for word, _ in graph_voc.word2idx.items():
            fout.write('{}\t{}\n'.format(word, str(level_dict[word])))


def generate_meta_for_not_graph(task, output_path='emb-meta.tsv'):
    tokenizer = EHRTokenizer(data_dir='../data')
    voc = tokenizer.dx_voc if task == 0 else tokenizer.rx_voc
    with open(os.path.join(output_dir, ('dx-' if task == 0 else 'rx-') + output_path), 'w') as fout:
        # fout.write('name\n')
        for word, _ in voc.word2idx.items():
            fout.write('{}\n'.format(word))


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save()

    # generate_meta_for_not_graph(task=0)
    # generate_meta_for_not_graph(task=1)

    generate_meta(build_atc_tree, task=1)
    generate_meta(build_icd9_tree, task=0)
