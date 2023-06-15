#!/usr/bin/python3

import json

import numpy as np
from torch.utils.data import Dataset, DataLoader


# TODO: Add unified tokens, expressing both relations, entities, and logical operations
#
special_token_dict = {
    "(": 0,
    ")": 1,
    "p": 2,
    "i": 3,
    "u": 4,
    "n": 5,
}
std = special_token_dict
std_offset = 100


def abstraction(instantiated_query, nentity, nrelation):

    query = instantiated_query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    def r(relation_id):
        return relation_id + nentity + std_offset

    def e(entity_id):
        return entity_id + std_offset

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    if sub_queries[0] == "p":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list =  [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"] , r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "e":

        entity_id = int(sub_queries[1][1:-1])

        ids_list = [entity_id]
        this_query_type = "(e)"
        this_unified_ids = [std["("], e(entity_id), std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "i":

        ids_list = []
        this_query_type = "(i"
        this_unified_ids = [std["("], std["i"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type +"," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids


    elif sub_queries[0] == "u":
        ids_list = []
        this_query_type = "(u"
        this_unified_ids = [std["("], std["u"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "n":
        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[1], nentity, nrelation)
        return sub_ids_list, "(n," + sub_query_type + ")", [std["("], std["n"]] + sub_unified_ids + [std[")"]]

    else:
        print("Invalid Pattern")
        exit()



class Instantiation(object):


    def __init__(self, value_matrix):
        self.value_matrix = np.array(value_matrix)


    def instantiate(self, query_pattern):

        query = query_pattern[1:-1]
        parenthesis_count = 0

        sub_queries = []
        jj = 0

        for ii, character in enumerate(query):
            # Skip the comma inside a parenthesis
            if character == "(":
                parenthesis_count += 1

            elif character == ")":
                parenthesis_count -= 1

            if parenthesis_count > 0:
                continue

            if character == ",":
                sub_queries.append(query[jj: ii])
                jj = ii + 1

        sub_queries.append(query[jj: len(query)])

        if sub_queries[0] == "p":


            relation_ids = self.value_matrix[:,0]
            self.value_matrix = self.value_matrix[:,1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return ("p", relation_ids, sub_batched_query )

        elif sub_queries[0] == "e":
            entity_ids = self.value_matrix[:,0]
            self.value_matrix = self.value_matrix[:, 1:]

            return  ("e",  entity_ids)

        elif sub_queries[0] == "i":

            return_list = ["i"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)


        elif sub_queries[0] == "u":
            return_list = ["u"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)

        elif sub_queries[0] == "n":
            sub_batched_query = self.instantiate(sub_queries[1])

            return ("n", sub_batched_query)

        else:
            print("Invalid Pattern")
            exit()



class TestDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []

        self.unified_id_list = []

        self.query_type = None

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            self.test_answer_list.append([int(ans) for ans in answer_list["test_answers"]])

            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        unified_id_list  = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, test_answer_list, self.query_type

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]
        test_answers = [_[4] for _ in data]

        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[5] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers, test_answers


class ValidDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.unified_id_list = []

        self.query_type = None

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])

            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)
            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, self.query_type

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]


        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[4] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers


# The design of the data loader is tricky. There are two requirements of loading. 1. We cannot do parsing of each sample
# during the collate time, or the training speed will be too slow. This is because we may use large batch size.
# 2. The data loader must be flexible enough to deal with all types of queries automatically. As for one data loader
# only deal with one type of query, we can store all the numerical values in a separate matrix, and memorize all their
# index. By doing this, we can fast reconstruct the structured and batched result quickly.


class TrainDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.answer_list = []
        self.query_type = None

        self.unified_id_list = []

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)
            self.answer_list.append([int(ans )for ans in answer_list["train_answers"]])
            self.id_list.append(this_id_list)

            self.unified_id_list.append(unified_ids)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        ids_in_query = self.id_list[idx]
        answer_list = self.answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        tail = np.random.choice(list(answer_list))

        positive_sample = int(tail)

        return ids_in_query, unified_id_list, positive_sample, self.query_type

    @staticmethod
    def collate_fn(data):
        positive_sample = [_[2] for _ in data]
        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[3] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, positive_sample




class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        self.len = len(dataloader)

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


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
        for  batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break







