from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from predictive_models import GBERT_Predict_Side


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # code only in multi-visit data
        self.rx_voc_multi = Voc()
        self.dx_voc_multi = Voc()
        with open(os.path.join(data_dir, 'rx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rx_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'dx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.dx_voc_multi.add_sentence([code.rstrip('\n')])

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


class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0
        self.side_len = len(self.data_pd.iloc[0, 5:])
        logger.info('side len %d' % self.side_len)

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            records = {}
            side_records = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient = []
                sides = []
                for _, row in item_df.iterrows():
                    admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
                    patient.append(admission)
                    sides.append(row[5:].values)
                if len(patient) < 2:
                    continue
                records[subject_id] = patient
                side_records[subject_id] = sides
            return records, side_records

        self.records, self.side_records = transform_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        subject_id = list(self.records.keys())[item]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)

        for idx, adm in enumerate(self.records[subject_id]):
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
            # output_rx_tokens.append(list(adm[1]))

            if idx != 0:
                output_rx_tokens.append(list(adm[1]))
                output_dx_tokens.append(list(adm[0]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)

        dx_voc_size = len(self.tokenizer.dx_voc_multi.word2idx)
        rx_voc_size = len(self.tokenizer.rx_voc_multi.word2idx)
        for tokens in output_dx_tokens:
            tmp_labels = np.zeros(dx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.dx_voc_multi.word2idx[x], tokens))] = 1
            output_dx_labels.append(tmp_labels)

        for tokens in output_rx_tokens:
            tmp_labels = np.zeros(rx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.rx_voc_multi.word2idx[x], tokens))] = 1
            output_rx_labels.append(tmp_labels)

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("subject_id: %s" % subject_id)
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        assert len(input_ids) == (self.seq_len *
                                  2 * len(self.records[subject_id]))
        assert len(output_dx_labels) == (len(self.records[subject_id]) - 1)
        # assert len(output_rx_labels) == len(self.records[subject_id])-1
        """extract side 
        """
        sides = self.side_records[subject_id][1:]
        assert len(sides) == len(output_dx_labels)

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_dx_labels, dtype=torch.float),
                       torch.tensor(output_rx_labels, dtype=torch.float),
                       torch.tensor(sides, dtype=torch.float))

        return cur_tensors


def load_dataset(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = pd.read_pickle(os.path.join(data_dir, 'data-multi-visit.pkl'))
    # load side
    side_pd = pd.read_pickle(os.path.join(data_dir, 'data-multi-side.pkl'))
    # concat
    data = data.merge(side_pd, how='inner', on=['SUBJECT_ID', 'HADM_ID'])

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'train-id.txt'),
                os.path.join(data_dir, 'eval-id.txt'),
                os.path.join(data_dir, 'test-id.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(int(line.rstrip('\n')))
        return data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)

    return tokenizer, tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-predict-side', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-predict', type=str, required=False,
                        help="pretraining model dir.")
    parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=True,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=55,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=40.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    tokenizer, (train_dataset, eval_dataset, test_dataset) = load_dataset(args)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=1)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=SequentialSampler(eval_dataset),
                                 batch_size=1)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=1)

    print('Loading Model: ' + args.model_name)
    # config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx), side_len=train_dataset.side_len)
    # config.graph = args.graph
    # model = SeperateBertTransModel(config, tokenizer.dx_voc, tokenizer.rx_voc)
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = GBERT_Predict_Side.from_pretrained(
            args.pretrain_dir, tokenizer=tokenizer, side_len=train_dataset.side_len)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = GBERT_Predict_Side(config, tokenizer, train_dataset.side_len)

    logger.info('# of model parameters: ' + str(get_n_params(model)))

    model.to(device)

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    rx_output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin")

    # Prepare optimizer
    # num_train_optimization_steps = int(
    #     len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", 1)

        dx_acc_best, rx_acc_best = 0, 0
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        rx_history = {'prauc': []}

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(prog_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, rx_labels, input_sides = batch
                input_ids, dx_labels, rx_labels, input_sides = input_ids.squeeze(
                    dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0), input_sides.squeeze(dim=0)
                loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels,
                                        epoch=global_step, input_sides=input_sides)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            global_step += 1

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                model.eval()
                rx_y_preds = []
                rx_y_trues = []
                for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
                    eval_input = tuple(t.to(device) for t in eval_input)
                    input_ids, dx_labels, rx_labels, input_sides = eval_input
                    input_ids, dx_labels, rx_labels, input_sides = input_ids.squeeze(
                    ), dx_labels.squeeze(), rx_labels.squeeze(dim=0), input_sides.squeeze(dim=0)
                    with torch.no_grad():
                        loss, rx_logits = model(
                            input_ids, dx_labels=dx_labels, rx_labels=rx_labels, input_sides=input_sides)
                        rx_y_preds.append(t2n(torch.sigmoid(rx_logits)))
                        rx_y_trues.append(t2n(rx_labels))

                print('')
                rx_acc_container = metric_report(np.concatenate(rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0),
                                                 args.therhold)
                writer.add_scalars(
                    'eval_rx', rx_acc_container, global_step)

                if rx_acc_container[acc_name] > rx_acc_best:
                    rx_acc_best = rx_acc_container[acc_name]
                    # save model
                    torch.save(model_to_save.state_dict(),
                               rx_output_model_file)

        with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
            fout.write(model.config.to_json_string())

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", 1)

        def test(task=0):
            # Load a trained model that you have fine-tuned
            model_state_dict = torch.load(rx_output_model_file)
            model.load_state_dict(model_state_dict)
            model.to(device)

            model.eval()
            y_preds = []
            y_trues = []
            for test_input in tqdm(test_dataloader, desc="Testing"):
                test_input = tuple(t.to(device) for t in test_input)
                input_ids, dx_labels, rx_labels, input_sides = test_input
                input_ids, dx_labels, rx_labels, input_sides = input_ids.squeeze(
                ), dx_labels.squeeze(), rx_labels.squeeze(dim=0), input_sides.squeeze(dim=0)
                with torch.no_grad():
                    loss, rx_logits = model(
                        input_ids, dx_labels=dx_labels, rx_labels=rx_labels, input_sides=input_sides)
                    y_preds.append(t2n(torch.sigmoid(rx_logits)))
                    y_trues.append(t2n(rx_labels))

            print('')
            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                          args.therhold)

            # save report
            writer.add_scalars('test', acc_container, 0)

            return acc_container

        test(task=0)


if __name__ == "__main__":
    main()
