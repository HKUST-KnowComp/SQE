import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss

# from .dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
# from .model import IterativeModel


class TreeLSTM(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False,negative_size=128):
        super(TreeLSTM, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

        self.operation_embedding = nn.Embedding(3, embedding_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.W_i = nn.Linear(embedding_size, embedding_size)
        self.U_i = nn.Linear(embedding_size, embedding_size)

        self.W_f = nn.Linear(embedding_size, embedding_size)
        self.U_f = nn.Linear(embedding_size, embedding_size)

        self.W_o = nn.Linear(embedding_size, embedding_size)
        self.U_o = nn.Linear(embedding_size, embedding_size)

        self.W_u = nn.Linear(embedding_size, embedding_size)
        self.U_u = nn.Linear(embedding_size, embedding_size)

        self.decoder = nn.Linear(embedding_size, num_entities)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        embedding_weights = self.entity_embedding.weight

        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights




    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        # print("query_encoding", query_encoding.shape)
        query_scores = self.decoder(query_encoding[:, 0, :])
        return query_scores


    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    def projection(self, relation_ids, sub_query_encoding):
        """
        The relational projection of GQE. To fairly evaluate results, we use the same size of relation and use
        TransE embedding structure.

        :param relation_ids: [batch_size]
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        # [batch_size, embedding_size]
        if len(sub_query_encoding.shape) == 2:
            prev_h = sub_query_encoding
            prev_c = torch.zeros_like(prev_h)

        else:
            prev_h = sub_query_encoding[:, 0, :]
            prev_c = sub_query_encoding[:, 1, :]



        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        x = self.relation_embedding(relation_ids)
        
        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))
        f = self.sigmoid(self.W_f(x) + self.U_f(prev_h))
        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))
        u = self.tanh(self.W_u(x) + self.U_u(prev_h))

        next_c = f * prev_c + i * u
        next_h = o * self.tanh(next_c)



        return torch.stack((next_h, next_c), dim=1)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, 2, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, embedding_size]
        prev_h = all_subquery_encodings[:, :, 0, :].sum(dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        c_k = all_subquery_encodings[:, :, 1, :]

        x = self.operation_embedding(
            torch.zeros(all_subquery_encodings.shape[0]).long().to(self.operation_embedding.weight.device)
        )

        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))

        # [batch_size, number_sub_queries, embedding_size]
        f_k = self.sigmoid( self.W_f(all_subquery_encodings[:, :, 0, :]) + self.U_f(prev_h).unsqueeze(1))

        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))

        u = self.tanh(self.W_u(x) + self.U_u(prev_h))


        next_c = torch.sum(f_k * c_k, dim=1) + i * u

        next_h = o * self.tanh(next_c)


        return torch.stack((next_h, next_c), dim=1)

    def negation(self, sub_query_encoding):
        # [batch_size, 2, embedding_size]

        prev_h = sub_query_encoding[:, 0, :]
        prev_c = sub_query_encoding[:, 1, :]

        operation_ids = torch.tensor(2).to(self.operation_embedding.weight.device).unsqueeze(0)
        x = self.operation_embedding(operation_ids)

        
        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))
        f = self.sigmoid(self.W_f(x) + self.U_f(prev_h))
        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))
        u = self.tanh(self.W_u(x) + self.U_u(prev_h))

        next_c = f * prev_c + i * u
        next_h = o * self.tanh(next_c)

        return torch.stack((next_h, next_c), dim=1)

    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, 2, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, embedding_size]
        prev_h = all_subquery_encodings[:, :, 0, :].sum(dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        c_k = all_subquery_encodings[:, :, 1, :]

        x = self.operation_embedding(
            torch.ones(all_subquery_encodings.shape[0]).long().to(self.operation_embedding.weight.device)
        )

        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))

        # [batch_size, number_sub_queries, embedding_size]
        f_k = self.sigmoid( self.W_f(all_subquery_encodings[:, :, 0, :]) + self.U_f(prev_h).unsqueeze(1))

        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))

        u = self.tanh(self.W_u(x) + self.U_u(prev_h))


        next_c = torch.sum(f_k * c_k, dim=1) + i * u

        next_h = o * self.tanh(next_c)


        return torch.stack((next_h, next_c), dim=1)



if __name__ == "__main__":

    sample_data_path = "../_sampled_data_same/"
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

    model = TreeLSTM(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():

        print("====================================")
        print(query_type)

        new_iterator = dataloader.SingledirectionalOneShotIterator(DataLoader(
            dataloader.TrainDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TrainDataset.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        batched_query, unified_ids, positive_sample = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

        query_embedding = model(batched_query)
        print(query_embedding.shape)
        loss = model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

    

        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            dataloader.ValidDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            query_embedding = model(batched_query)
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
            dataloader.TestDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TestDataset.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)

            query_embedding = model(batched_query)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
