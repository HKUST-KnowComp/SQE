import json
from pickle import FALSE

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator


class GeneralModel(nn.Module):

    def __init__(self, num_entities, num_relations, embedding_size):
        super(GeneralModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size

    def loss_fnt(self, sub_query_encoding, labels):
        raise NotImplementedError

    def scoring(self, query_encoding):
        """
        :param query_encoding:
        :return: [batch_size, num_entities]
        """
        raise NotImplementedError

    def evaluate_entailment(self, query_encoding, entailed_answers):
        """
        We do not have to conduct the evaluation on GPU as it is not necessary.


        :param query_encoding:
        :param entailed_answers:
        :return:
        """

        # [batch_size, num_entities]
        all_scoring = self.scoring(query_encoding)

        # [batch_size, num_entities]
        original_scores = all_scoring.clone()

        log_list = []

        for i in range(len(entailed_answers)):
            entailed_answer_set = torch.tensor(entailed_answers[i])

            # [num_entities]
            not_answer_scores = all_scoring[i]
            not_answer_scores[entailed_answer_set] = - 10000000

            # [1, num_entities]
            not_answer_scores = not_answer_scores.unsqueeze(0)

            # [num_entailed_answers, 1]
            entailed_answers_scores = original_scores[i][entailed_answer_set].unsqueeze(1)

            # [num_entailed_answers, num_entities]
            answer_is_smaller_matrix = ((entailed_answers_scores - not_answer_scores) < 0)

            # [num_entailed_answers, num_entities]
            answer_is_equal_matrix = ((entailed_answers_scores - not_answer_scores) == 0)

            # [num_entailed_answer]
            answer_tied_num = answer_is_equal_matrix.sum(dim = -1)

            # [num_entailed_answer]
            random_tied_addition = torch.mul(torch.rand(answer_tied_num.size()).to(answer_tied_num.device), answer_tied_num).type(torch.int64)

            # [num_entailed_answers]
            answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

            # [num_entailed_answers]
            answer_rankings = torch.add(answer_rankings, random_tied_addition)

            # [num_entailed_answers]
            rankings = answer_rankings.float()

            mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
            hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
            hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
            hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

            num_answers = len(entailed_answers[i])

            logs = {
                "ent_mrr": mrr,
                "ent_hit_at_1": hit_at_1,
                "ent_hit_at_3": hit_at_3,
                "ent_hit_at_10": hit_at_10,
                "ent_num_answers": num_answers
            }

            log_list.append(logs)
        return log_list

    def evaluate_generalization(self, query_encoding, entailed_answers, generalized_answers):
        """

        This function is largely different from the evaluation of previous work, and we conduct a more rigorous
        evaluation. In previous methods, when it contains existential positive queries without negations, we can all
        the answers in the graph that are entailed ``easy'' answers. But when it includes the negation operations,
        the answers entailed by the training graph may not be valid answers anymore !!! This is very critical in terms
        of evaluating the queries with negation/exclusion , but is ignored in all relevant work. Without evaluating
        the answers that are excluded by the reasoning, we cannot evaluate the actual performance of complement.

        :param query_encoding:
        :param entailed_answers:
        :param generalized_answers:
        :return:
        """

        # [batch_size, num_entities]
        all_scoring = self.scoring(query_encoding)

        # [batch_size, num_entities]
        original_scores = all_scoring.clone()

        log_list = []

        for i in range(len(entailed_answers)):

            all_answers = list(set(entailed_answers[i]) | set(generalized_answers[i]))
            need_to_exclude_answers = list(set(entailed_answers[i]) - set(generalized_answers[i]))
            need_to_inferred_answers = list(set(generalized_answers[i]) - set(entailed_answers[i]))

            all_answers_set = torch.tensor(all_answers)

            # [num_entities]
            not_answer_scores = all_scoring[i]
            not_answer_scores[all_answers_set] = - 10000000

            # [1, num_entities]
            not_answer_scores = not_answer_scores.unsqueeze(0)

            logs = {}

            if len(need_to_inferred_answers) > 0:
                num_answers = len(need_to_inferred_answers)

                need_to_inferred_answers = torch.tensor(need_to_inferred_answers)

                # [num_need_to_inferred_answers, 1]
                need_to_inferred_answers_scores = original_scores[i][need_to_inferred_answers].unsqueeze(1)

                # [num_need_to_inferred_answers, num_entities]
                answer_is_smaller_matrix = ((need_to_inferred_answers_scores - not_answer_scores) < 0)

                # [num_need_to_inferred_answers, num_entities]
                answer_is_equal_matrix = ((need_to_inferred_answers_scores - not_answer_scores) == 0)

                # [num_need_to_inferred_answers]
                answer_tied_num = answer_is_equal_matrix.sum(dim = -1)

                # [num_need_to_inferred_answers]
                random_tied_addition = torch.mul(torch.rand(answer_tied_num.size()).to(answer_tied_num.device), answer_tied_num).type(torch.int64)

                # [num_need_to_inferred_answers]
                answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                # [num_need_to_inferred_answers]
                answer_rankings = torch.add(answer_rankings, random_tied_addition)

                # [num_need_to_inferred_answers]
                rankings = answer_rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                logs["inf_mrr"] = mrr
                logs["inf_hit_at_1"] = hit_at_1
                logs["inf_hit_at_3"] = hit_at_3
                logs["inf_hit_at_10"] = hit_at_10
                logs["inf_num_answers"] = num_answers
            else:
                logs["inf_mrr"] = 0
                logs["inf_hit_at_1"] = 0
                logs["inf_hit_at_3"] = 0
                logs["inf_hit_at_10"] = 0
                logs["inf_num_answers"] = 0

            if len(need_to_exclude_answers) > 0:
                num_answers = len(need_to_exclude_answers)

                need_to_exclude_answers = torch.tensor(need_to_exclude_answers)

                # [num_need_to_exclude_answers, 1]
                need_to_exclude_answers_scores = original_scores[i][need_to_exclude_answers].unsqueeze(1)

                # [num_need_to_exclude_answers, num_entities]
                answer_is_smaller_matrix = ((need_to_exclude_answers_scores - not_answer_scores) < 0)

                # [num_need_to_exclude_answers, num_entities]
                answer_is_equal_matrix = ((need_to_exclude_answers_scores - not_answer_scores) == 0)

                # [num_need_to_exclude_answers]
                answer_tied_num = answer_is_equal_matrix.sum(dim = -1)

                # [num_need_to_exclude_answers]
                random_tied_addition = torch.mul(torch.rand(answer_tied_num.size()).to(answer_tied_num.device), answer_tied_num).type(torch.int64)

                # [num_need_to_exclude_answers]
                answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                # [num_need_to_exclude_answers]
                answer_rankings = torch.add(answer_rankings, random_tied_addition)

                # [num_need_to_exclude_answers]
                rankings = answer_rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                logs["exd_mrr"] = mrr
                logs["exd_hit_at_1"] = hit_at_1
                logs["exd_hit_at_3"] = hit_at_3
                logs["exd_hit_at_10"] = hit_at_10
                logs["exd_num_answers"] = num_answers
            else:
                logs["exd_mrr"] = 0
                logs["exd_hit_at_1"] = 0
                logs["exd_hit_at_3"] = 0
                logs["exd_hit_at_10"] = 0
                logs["exd_num_answers"] = 0

            log_list.append(logs)
        return log_list


class IterativeModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size, use_old_loss = False):
        super(IterativeModel, self).__init__(num_entities, num_relations, embedding_size)
        self.use_old_loss = use_old_loss


    def projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def higher_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def intersection(self, sub_query_encoding_list):
        raise NotImplementedError

    def union(self, sub_query_encoding_list):
        raise NotImplementedError

    def negation(self, sub_query_encoding):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):

        assert batched_structured_query[0] in ["p", "e", "i", "u", "n"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.intersection(sub_query_result_list)

        elif batched_structured_query[0] == "u":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.union(sub_query_result_list)

        elif batched_structured_query[0] == "n":
            sub_query_result = self.forward(batched_structured_query[1])
            this_query_result = self.negation(sub_query_result)

        elif batched_structured_query[0] == "e":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_result = self.entity_embedding(entity_ids)

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            if self.use_old_loss == False:
                return self.loss_fnt(this_query_result, label)
            else:
                return self.old_loss_fnt(this_query_result, label)


class SequentialModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size):
        super().__init__(num_entities, num_relations, embedding_size)

    def encode(self, batched_structured_query):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):
        
        batched_structured_query = torch.tensor(batched_structured_query)
        if torch.cuda.is_available():
            batched_structured_query = batched_structured_query.cuda()

        representations = self.encode(batched_structured_query)

        if label is not None:
            return self.loss_fnt(representations, label)

        else:
            return representations


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


if __name__ == "__main__":

    train_data_path = "./FB15k-237-betae_train_queries.json"
    valid_data_path = "./FB15k-237-betae_valid_queries.json"
    test_data_path = "./FB15k-237-betae_test_queries.json"
    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_path = "./KG_data/FB15k-237-betae"

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

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
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
