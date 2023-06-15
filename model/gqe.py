import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss

# from .dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
# from .model import IterativeModel


class GQE(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False,negative_size=128):
        super(GQE, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

        self.intersection_nn_layer_1 = nn.Linear(embedding_size, embedding_size // 2)
        self.relu = nn.ReLU()
        self.intersection_nn_layer_2 = nn.Linear(embedding_size // 2, embedding_size)
        self.intersection_transformation_matrix = nn.Linear(embedding_size, embedding_size, bias=False)

        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.use_old_loss = use_old_loss
        self.negative_size = negative_size
    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        # TODO: fix the scoring function here, this function is not correct
        query_scores = self.decoder(query_encoding)
        return query_scores

    def old_loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, embedding_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)
        label_entity_embeddings = self.entity_embedding(labels)

        batch_size = label_entity_embeddings.shape[0]
        # [batch_size]
        random_entity_indices = torch.randint(0, self.num_entities, (self.negative_size,batch_size)).to(
            self.entity_embedding.weight.device)

        # [batch_size, embedding_size]
        negative_samples_embeddings = self.entity_embedding(random_entity_indices)

        query_encoding_norm = torch.sqrt(torch.sum(query_encoding * query_encoding, dim=-1))
        positive_embedding_norm = torch.sqrt(torch.sum(label_entity_embeddings * label_entity_embeddings, dim=-1))
        negative_embedding_norm = torch.sqrt(
            torch.sum(negative_samples_embeddings * negative_samples_embeddings, dim=-1))

        # [batch_size]
        positive_scores = torch.sum(query_encoding * label_entity_embeddings, dim=-1) / \
                          query_encoding_norm / positive_embedding_norm

        # [batch_size]
        negative_scores = torch.sum(query_encoding * negative_samples_embeddings, dim=-1) / \
                          query_encoding_norm / negative_embedding_norm

        relu = nn.ReLU()
        loss = torch.mean(relu(1 + negative_scores - positive_scores))
        
        return loss

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
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        relation_embeddings = self.relation_embedding(relation_ids)

        return relation_embeddings + sub_query_encoding

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)

    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        all_subquery_encodings = self.intersection_nn_layer_1(all_subquery_encodings)
        all_subquery_encodings = self.relu(all_subquery_encodings)
        all_subquery_encodings = self.intersection_nn_layer_2(all_subquery_encodings)

        # The implementation of \phi is mean pooling
        # [batch_size, embedding_size]
        all_subquery_encodings = torch.mean(all_subquery_encodings, dim=1)

        # The transformation matrix
        # [batch_size, embedding_size]
        all_subquery_encodings = self.intersection_transformation_matrix(all_subquery_encodings)

        return all_subquery_encodings


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

    gqe_model = GQE(num_entities=nentity, num_relations=nrelation, embedding_size=300,use_old_loss=True)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

        query_embedding = gqe_model(batched_query)
        print(query_embedding.shape)
        loss = gqe_model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
