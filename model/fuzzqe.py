import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss



class FuzzQE(IterativeModel):
    
    def __init__(self, num_entities, num_relations, embedding_size = 500, K = 30, label_smoothing=0.0):
        super(FuzzQE, self).__init__(num_entities, num_relations, embedding_size)


        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=0, 
            b=1
        )
        self.relation_coefficient = nn.Embedding(num_relations, K) #alpha_rj, (j = 1, ..., K, r = 1, ..., num_relations)
        nn.init.uniform_( #not mentioned in the original paper, here we try to use uniform distribution
            tensor=self.relation_coefficient.weight, 
            a=0,
            b=1
        )
        self.relation_matrix = nn.Embedding(K, embedding_dim=embedding_size*embedding_size) # K matrices of size D * D
        self.relation_vector = nn.Embedding(K, embedding_size) # K vectors of size D
        self.K = K

        
        self.embedding_size = embedding_size #hyper-parameter 

        self.layer_norm = nn.LayerNorm(embedding_size)   

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)




    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """
        entity_embeddings = self.entity_embedding.weight
         
        scores = torch.matmul(entity_embeddings,query_encoding.t()).t() # [batch_size, num_entities]
        #print(scores[0]) 
        # print("scores", scores.shape, scores)
        
        return scores

    def loss_fnt(self, sub_query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(sub_query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss


    def projection(self, relation_ids, sub_query_encoding):
        """
        The relational projection of FuzzQE. 
        Sq = g ( LN ( WrPe + br) )
        """
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)


        relation_ids = relation_ids.to(self.relation_vector.weight.device)
        alpha = self.relation_coefficient(relation_ids) # [batch_size, K]

        M = self.relation_matrix.weight.reshape([self.K, self.embedding_size, self.embedding_size]) # [K, embedding_size, embedding_size]
        V = self.relation_vector.weight # [K, embedding_size]
        # W_r = sum_j (alpha_[j] * M[j])
        '''
        alpha_enlarged_matrix = alpha.repeat([self.embedding_size,self.embedding_size,1,1]).permute(2,3,1,0) # [batch_size, K, embedding_size, embedding_size]'''
        alpha_enlarged_vector = alpha.repeat([self.embedding_size,1,1]).permute(1,2,0) # [batch_size, K, embedding_size]

        #we don't use the enlarged method for M, instead we use a forloop to achieve same result
        W_r = torch.einsum('ab,bjk->ajk',alpha,M)
        b_r = torch.einsum('ab,bj->aj',alpha,V)
        '''
        for i in range(alpha.shape[0]):
            b = torch.einsum('b,bjk->jk',alpha[i],M)
            if i == 0:
                W_r = b.unsqueeze(0).to(self.entity_embedding.weight.device)
            else:
                W_r = torch.cat((W_r,b.unsqueeze(0)),dim = 0) # [batch_size, embedding_size, embedding_size]'''


        #W_r = torch.sum(alpha_enlarged_matrix * M, dim=1) # [batch_size, embedding_size, embedding_size]
        #b_r = torch.sum(alpha_enlarged_vector * V, dim=1) # [batch_size, embedding_size]

        # weight * encoding + bias
        #query_encoding = torch.matmul(sum_matrix,batched_qe.unsqueeze(2)).squeeze() + sum_v
        query_embedding = torch.matmul (W_r,sub_query_encoding.unsqueeze(2)).squeeze() + b_r # [batch_size, embedding_size]
        
        query_embedding = self.layer_norm(query_embedding) # layer normalization

        query_embedding = torch.special.expit(query_embedding) #logistic sigmoid function

        return query_embedding

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)

    def union(self, sub_query_encoding_list):
        """
        q1 + q2 - (q1 * q2)
        """
  
        sub_query_encoding_1 = sub_query_encoding_list[0]
        sub_query_encoding_2 = sub_query_encoding_list[1]
        #sum of the two sub-query embeddings minus element-wise product of the two sub-query embeddings
        union_query_embedding = sub_query_encoding_1 + sub_query_encoding_2 - (sub_query_encoding_1 * sub_query_encoding_2)
        
        return union_query_embedding

    def intersection(self, sub_query_encoding_list):
        """
        q1 * q2
        """
        sub_query_encoding_1 = sub_query_encoding_list[0]
        sub_query_encoding_2 = sub_query_encoding_list[1]
        all_subquery_encodings = sub_query_encoding_1 * sub_query_encoding_2
        return all_subquery_encodings
    
    def negation(self, query_encoding):
        #1 - q
        one_tensor = torch.ones(query_encoding.size()).to(self.entity_embedding.weight.device)
        negation_query_encoding =  one_tensor - query_encoding
        return negation_query_encoding

    #Override Forward Function: adding regularizer to "entity"
    #embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
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
            raw_entity_embedding = self.entity_embedding(entity_ids)
            this_query_result = raw_entity_embedding

        else:
            this_query_result = None
        if label is None:
            
            return this_query_result

        else:

            return self.loss_fnt(this_query_result, label)

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

    fuzzqe_model =FuzzQE(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    #if torch.cuda.is_available():
        #fuzzqe_model = fuzzqe_model.cuda()

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

        query_embedding = fuzzqe_model(batched_query)
        print(query_embedding.shape)
        loss = fuzzqe_model(batched_query, positive_sample)
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

            query_embedding = fuzzqe_model(batched_query)
            result_logs = fuzzqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = fuzzqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
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

            query_embedding = fuzzqe_model(batched_query)
            result_logs = fuzzqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = fuzzqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)


            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
