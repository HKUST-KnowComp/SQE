import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class Q2B(IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, gamma=12, alpha=0.02, label_smoothing=0.1,use_old_loss=False,negative_size = 128):
        super(Q2B, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

        # initialize embeddings
        # we treat entities as boxes with offset=0
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        self.relation_center_embedding = nn.Embedding(num_relations, embedding_size)
        self.relation_offset_embedding = nn.Embedding(num_relations, embedding_size)
        self.embedding_size = embedding_size
        self.negative_size = negative_size
        # box intersection nets

        self.center_intersection_net = CenterIntersection(self.embedding_size)
        self.offset_intersection_net = BoxOffsetIntersection(self.embedding_size)

        self.gamma = nn.Parameter(  # Margin when calculating score
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.alpha = nn.Parameter(  # ratio to discount in-box distance
            torch.Tensor([alpha]),
            requires_grad=False
        )

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.use_old_loss = use_old_loss


    def distance_box(self, entity_embedding, query_box_embedding):
        # query_box_embedding: tuple[query_center_embedding, query_offset_embedding]
        query_center_embedding, query_offset_embedding = query_box_embedding
        # calculate distance between entity and query box
        raw_distance = torch.sub(entity_embedding, query_center_embedding).abs()
        distance_out = F.relu(raw_distance - query_offset_embedding)
        distance_in = torch.min(raw_distance, query_offset_embedding)
        distance = distance_out + self.alpha * distance_in

        return distance

    def scoring(self, query_box_encoding):
        """

        :param query_box_encoding: ([batch_size, embedding_size],[batch_size, embedding_size])
        :return: [batch_size, num_entities]
        """

        # TODO: Fix the training process of the query2box. Do not use the triplet as it is extremely not robust and
        #  heavily relies on the hyper-parameter tuning

        # [num_entities, embedding_size]
        entity_embeddings = self.entity_embedding.weight
        query_center_embedding, query_offset_embedding = query_box_encoding

        # we need distances to be [batch_size, num_entities]

        # [1, num_entities, embedding_size]
        enlarged_entity_embeddings = entity_embeddings.unsqueeze(0)

        # [batch_size, 1, embedding_size]
        enlarged_center_embeddings = query_center_embedding.unsqueeze(1)

        # [batch_size, 1, embedding_size]
        enlarged_offset_embeddings = query_offset_embedding.unsqueeze(1)

        q_max = enlarged_center_embeddings + enlarged_offset_embeddings
        q_min = enlarged_center_embeddings - enlarged_offset_embeddings

        # [batch_size, num_entities]
        dist_out = (F.relu(enlarged_entity_embeddings - q_max) + F.relu(q_min - enlarged_entity_embeddings)).sum(dim=-1)

        dist_in = (enlarged_center_embeddings - torch.minimum( q_max, torch.maximum(q_min, enlarged_entity_embeddings))).abs().sum(dim=-1)

        # # [batch_size, num_entities, embedding_size]
        # raw_distance = torch.sub(enlarged_entity_embeddings, enlarged_center_embeddings).abs()
        #
        # # [batch_size, num_entities]
        # distance_out = F.relu(raw_distance - enlarged_offset_embeddings).sum(-1)
        # distance_in = torch.min(raw_distance, enlarged_offset_embeddings).sum(-1)
        distances = dist_out + self.alpha * dist_in
        
        return self.gamma - distances

    def projection(self, relation_ids, sub_query_box_embedding):
        """
        The relational projection of query2box

        :param relation_ids: [batch_size]
        :param sub_query_center_embedding: [batch_size, embedding_size]
        :param sub_query_offset_embedding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size], [batch_size, embedding_size] (center + offset)
        """
        # print(len(sub_query_box_embedding))
        sub_query_center_embedding, sub_query_offset_embedding = sub_query_box_embedding
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(
            self.relation_center_embedding.weight.device)  # What's the usage of this sentence?

        relation_center_embeddings = self.relation_center_embedding(relation_ids)
        relation_offset_embeddings = self.relation_offset_embedding(relation_ids)

        new_center_embedding = relation_center_embeddings + sub_query_center_embedding
        new_offset_embedding = relation_offset_embeddings + sub_query_offset_embedding
        new_box_embedding = tuple([new_center_embedding, new_offset_embedding])

        return new_box_embedding

    def higher_projection(self, relation_ids, sub_query_box_embedding):
        return self.projection(relation_ids, sub_query_box_embedding)

    def intersection(self, sub_query_box_embedding_list):
        """
        :param: sub_query_box_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
        :return:  [batch_size, embedding_size], [batch_size, embedding_size]
        """
        sub_query_center_embedding_list, sub_query_offset_embedding_list = sub_query_box_embedding_list
        # num_sub_queries固定吗
        # 过完net之后的形状

        """
        [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]

        after intersection

        [a12,b12,c12,d12,e12] 
        """
        all_subquery_center_encodings = torch.stack(sub_query_center_embedding_list, dim=0)
        all_subquery_offset_encodings = torch.stack(sub_query_offset_embedding_list, dim=0)

        new_query_center_embeddings = self.center_intersection_net(all_subquery_center_encodings)
        new_query_offset_embeddings = self.offset_intersection_net(all_subquery_offset_encodings)

        new_query_box_embeddings = tuple([new_query_center_embeddings, new_query_offset_embeddings])
        return new_query_box_embeddings

    def old_loss_fnt(self, query_box_encoding, labels):

        # The size of the query_box_encoding is ([batch_size, embedding_size],[batch_size, embedding_size]) 
        # Consists of Center + Offset

        # and the labels are [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)
        label_entity_embeddings = self.entity_embedding(labels)

        batch_size = label_entity_embeddings.shape[0]
        # [batch_size]
        random_entity_indices = torch.randint(0, self.num_entities, (self.negative_size,batch_size)).to(
            self.entity_embedding.weight.device)

        # [batch_size, embedding_size]
        negative_samples_embeddings = self.entity_embedding(random_entity_indices)


        # [batch_size, embedding_size]
        positive_distance = self.distance_box(label_entity_embeddings, query_box_encoding)
        negative_distance = self.distance_box(negative_samples_embeddings, query_box_encoding)

        # Original Repo Normalized logits inside logsigmoid
        margin = self.gamma
        positive_score = -F.logsigmoid(margin - positive_distance)
        # print(positive_score.shape)
        negative_score = -F.logsigmoid(negative_distance - margin).mean(dim=0)

        # print(negative_score.shape)
        relu = nn.ReLU()
        loss = torch.mean(relu(positive_score + negative_score))
        return loss

    def loss_fnt(self, sub_query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(sub_query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    def forward(self, batched_structured_query, label=None):
        # We need to override this forward function as the structure of box embedding is different
        # input: batched_structured_query
        # output: BOX TUPLE instead of single embedding
        assert batched_structured_query[0] in ["p", "e", "i"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            # intersection of box embedding takes tuple of two lists
            sub_query_center_result_list = []
            sub_query_offset_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_center_result, sub_query_offset_result = self.forward(batched_structured_query[_i])
                sub_query_center_result_list.append(sub_query_center_result)
                sub_query_offset_result_list.append(sub_query_offset_result)

            sub_query_box_result_list = tuple([sub_query_center_result_list, sub_query_offset_result_list])
            this_query_result = self.intersection(sub_query_box_result_list)

        elif batched_structured_query[0] == "e":
            # set the offset tensor to all zeros
            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_center_result = self.entity_embedding(entity_ids)
            this_query_offset_result = torch.zeros(this_query_center_result.shape).to(
                self.entity_embedding.weight.device)
            this_query_result = tuple([this_query_center_result, this_query_offset_result])

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            if self.use_old_loss == False:
                return self.loss_fnt(this_query_result, label)
            else:
                return self.old_loss_fnt(this_query_result, label)


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

    q2b_model = Q2B(num_entities=nentity, num_relations=nrelation, embedding_size=300, use_old_loss=True)
    if torch.cuda.is_available():
        q2b_model = q2b_model.cuda()

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

        query_embedding = q2b_model(batched_query)
        print("center dimension:", query_embedding[0].shape)
        print("offset dimension:", query_embedding[1].shape)
        print(len(positive_sample))

        loss = q2b_model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

            query_embedding = q2b_model(batched_query)
            result_logs = q2b_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = q2b_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

            query_embedding = q2b_model(batched_query)
            result_logs = q2b_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = q2b_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
