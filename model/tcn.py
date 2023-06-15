import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator, std_offset, \
    special_token_dict
from model import SequentialModel, LabelSmoothingLoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig


class TemporalConvolutionLayer(nn.Module):
    def __init__(self, embedding_size, kernel_size, dropout=0.2):

        """
        :param in_channels: int
        :param out_channels: int
        :param kernel_size: int
        :param dropout: float
        
        The temporal convolution layer is a 2D convolution layer with a kernel size of (kernel_size, embedding_size).
        Kernel size will be largely chosen from 5 or 7.

        """
        super(TemporalConvolutionLayer, self).__init__()

        self.embedding_size = embedding_size
        self.conv2d = nn.Conv2d(1, embedding_size, (kernel_size, embedding_size), padding=(kernel_size // 2, 0))
        self._1x1con2d = nn.Conv2d(embedding_size, embedding_size, (1, 1), padding=(0, 0))
        
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, in_channels)
        :return: (batch_size, seq_len, out_channels)"""


        conv_x = x.unsqueeze(1)  # (batch_size, 1, seq_len, in_channels)
        conv_x = self.conv2d(conv_x)  # (batch_size, out_channels/embedding_size, seq_len, 1)
        conv_x = self.relu(conv_x) # (batch_size, out_channels/embedding_size, seq_len, 1)
        conv_x = self._1x1con2d(conv_x)  # (batch_size, out_channels, seq_len, 1)
        conv_x = conv_x.squeeze(3)  # (batch_size, out_channels/embedding_size, seq_len)
        conv_x = conv_x.transpose(1, 2)  # (batch_size, seq_len, out_channels/embedding_size)

        
        x = conv_x + x
        x = self.dropout(x)

        return x
        
class TCNModel(SequentialModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, seq_length=47):#need to set different seq_length for different task?
        super(TCNModel, self).__init__(num_entities, num_relations, embedding_size)

        self.unified_embeddings = nn.Embedding(num_entities + num_relations + std_offset, embedding_size)
        self.num_entities = num_entities
        embedding_weights = self.unified_embeddings.weight

        self.decoder = nn.Linear(embedding_size,
                                 num_entities + num_relations + std_offset, bias=False)

        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
       
        
        self.seq_length = seq_length
        self.num_kernels = 100
        self.kernel_sizes = [3,3,5,5,7,7]
        self.embedding_size = embedding_size

        self.convs = nn.ModuleList([TemporalConvolutionLayer(embedding_size=embedding_size, kernel_size=K) for K in self.kernel_sizes])



    def encode(self, batched_structured_query):
        """
        :param batched_structured_query: (batch_size, seq_len)
        :return: (batch_size, embedding_size)
        """
        # [batch_size, seq_len, embedding_size]
        embeddings = self.unified_embeddings(batched_structured_query)

        # [batch_size, seq_len, embedding_size]
        for conv in self.convs:
            embeddings = conv(embeddings)

        # [batch_size, embedding_size]
        query_encoding = torch.mean(embeddings, dim=1)


        return query_encoding

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

    model = TCNModel(num_entities=nentity, num_relations=nrelation, embedding_size=300)
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
