import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from yaml import BlockSequenceStartToken

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator, std_offset, \
    special_token_dict
from model import SequentialModel, LabelSmoothingLoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig

def generate_position_id(batched_structured_query):

    position_ids = torch.zeros(batched_structured_query.shape[-1]).cuda()
    level = 0
    query = batched_structured_query[0]
    for i in range(batched_structured_query.shape[-1]):
        if query[i]==0:
            level+=1
        elif query[i]==1:
            level-=1
        elif query[i]==3:
            level-=1 #intersection will bring extra "("s but might not add to positional encoding
        elif query[i]>=100:
            position_ids[i]=level

    return position_ids.repeat(batched_structured_query.shape[0],1).int()

def generate_mask_id(batched_structured_query):#1 for retained calculation, 0 for masked

    mask_ids = torch.zeros(batched_structured_query.shape[-1]).cuda()
    query = batched_structured_query[0]
    for i in range(batched_structured_query.shape[-1]):
        if query[i] >= 100:
            mask_ids[i]=1


    return mask_ids.repeat(batched_structured_query.shape[0],1)


class BiQEModel(SequentialModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, num_layers=3):
        super(BiQEModel, self).__init__(num_entities, num_relations, embedding_size)

        config = BertConfig(hidden_size=embedding_size, vocab_size=num_entities + num_relations + std_offset,
                            num_attention_heads=16, num_hidden_layers=num_layers)
        self.transformer_encoder = BertModel(config)

        embedding_weights = self.transformer_encoder.embeddings.word_embeddings.weight
        self.decoder = nn.Linear(self.transformer_encoder.config.hidden_size,
                                 self.transformer_encoder.config.vocab_size, bias=False)
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

    def encode(self, batched_structured_query):
        
        encoded = self.transformer_encoder(batched_structured_query, 
                                            position_ids = generate_position_id(batched_structured_query),
                                            attention_mask = generate_mask_id(batched_structured_query)).pooler_output

        return encoded

    def scoring(self, query_encoding):
        # [batch_size, num_entities]
        query_scores = self.decoder(query_encoding)[:, std_offset:std_offset + self.num_entities + 1]
        return query_scores
    
    def forward(self, batched_structured_query, label=None):
        
        batched_structured_query = torch.tensor(batched_structured_query)
        if torch.cuda.is_available():
            batched_structured_query = batched_structured_query.cuda()
        

        representations = self.encode(batched_structured_query)

        if label is not None:
            return self.loss_fnt(representations, label)

        else:
            return representations

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

    model = BiQEModel(num_entities=nentity, num_relations=nrelation, embedding_size=256)
    if torch.cuda.is_available():
        model = model.cuda()

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

        query_embedding = model(unified_ids)
        print(query_embedding.shape)
        loss = model(unified_ids, positive_sample)
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

            query_embedding = model(unified_ids)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)
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
