import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator, std_offset, \
    special_token_dict
from model import SequentialModel, LabelSmoothingLoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig


class RNNModel(SequentialModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, num_layers=3):
        super(RNNModel, self).__init__(num_entities, num_relations, embedding_size)

        self.unified_embeddings = nn.Embedding(num_entities + num_relations + std_offset, embedding_size)

        embedding_weights = self.unified_embeddings.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities + num_relations + std_offset, bias=False)
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        self.RNN = nn.RNN(input_size=embedding_size, hidden_size=embedding_size // 2, num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

    def encode(self, batched_structured_query):
        embeddings = self.unified_embeddings(batched_structured_query)

        # [batch_size, length, embedding_size] -> [batch_size, hidden_size]
        rnn_output, _ = self.RNN(embeddings)
        rnn_output = rnn_output[:, 0, :]

        return rnn_output

    def scoring(self, query_encoding):

        # [batch_size, num_entities]
        query_scores = self.decoder(query_encoding)[:, std_offset:std_offset + self.num_entities + 1]
        return query_scores

    def loss_fnt(self, query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(query_encoding.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss


if __name__ == "__main__":

    sample_data_path = "../sampled_data_same/"
    KG_data_path = "../KG_data/"

    train_data_path = sample_data_path + "FB15k-237-betae_train_queries_0.json"
    valid_data_path = sample_data_path + "FB15k-237-betae_valid_queries_0.json"
    test_data_path = sample_data_path + "FB15k-237-betae_test_queries_0.json"
    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_path = KG_data_path + "FB15k-237-betae"

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    model = RNNModel(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        batched_query, unified_ids, positive_sample = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

        query_embedding = model(unified_ids)
        print(query_embedding.shape)
        loss = model(unified_ids, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():
        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            ValidDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            query_embedding = model(unified_ids)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            TestDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TestDataset.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)

            query_embedding = model(unified_ids)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
