import json
from turtle import forward

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss


from functools import reduce


from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator




class MLPMixer(nn.Module):
    def __init__(self, embedding_size, num_patches = 4):
        super(MLPMixer, self).__init__()
    
        self.embedding_size = embedding_size

        self.num_patches = num_patches

        self.mlp1 = MlpOperatorSingle(2 * self.embedding_size // self.num_patches)
        self.mlp2 = MlpOperatorSingle(self.embedding_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.mlp3 = MlpOperator(self.embedding_size)


    def forward(self, x, y):

        # [batch_size, 2, embedding_size]
        input_tensor = torch.stack((x, y), dim=1)

        activations = input_tensor.view(-1, 2, self.num_patches, self.embedding_size // self.num_patches)

        activations = activations.permute(0, 2, 1, 3)

        activations = activations.reshape(-1, self.num_patches, 2 * self.embedding_size // self.num_patches)

        activations = self.mlp1(activations)

        activations = activations.reshape(-1, self.num_patches, 2,  self.embedding_size // self.num_patches)

        activations = activations.permute(0, 2, 1, 3)

        activations = activations.reshape(-1, 2, self.embedding_size)

        activations = activations + input_tensor

        normed_activations = self.layer_norm(activations)

        normed_activations = self.mlp2(normed_activations)

        normed_activations = normed_activations + activations


        return self.mlp3(normed_activations[:,0,:], normed_activations[:,1,:])



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias




class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """

    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class MlpOperator(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()


        self.mlp_layer_1 = nn.Linear(embedding_size * 2, embedding_size)
        self.mlp_layer_2 = nn.Linear(embedding_size, embedding_size // 2)
        self.mlp_layer_3 = nn.Linear(embedding_size //2, embedding_size)

        self.activation = nn.GELU()
    

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = self.activation(self.mlp_layer_1(x))
        x = self.activation(self.mlp_layer_2(x))
        x = self.mlp_layer_3(x)
        return x


class MlpOperatorSingle(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()


        self.mlp_layer_1 = nn.Linear(embedding_size, embedding_size // 2)
        self.mlp_layer_2 = nn.Linear(embedding_size // 2, embedding_size // 2)
        self.mlp_layer_3 = nn.Linear(embedding_size //2, embedding_size)

        self.activation = nn.GELU()
    

    def forward(self, x):
        x = self.activation(self.mlp_layer_1(x))
        x = self.activation(self.mlp_layer_2(x))
        x = self.mlp_layer_3(x)
        return x



class MLPReasoner(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, 
                 dropout_rate=0.3,
             value_vocab=None, ):
        super(MLPReasoner, self).__init__(num_entities, num_relations, embedding_size)


        self.num_entities = num_entities
        self.num_relations = num_relations

    
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
       
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

    
        self.value_vocab = value_vocab

      

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()



        

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

     
        
        
        # MLP operations

        self.projection_mlp = MlpOperator(embedding_size)
        self.union_mlp = MlpOperator(embedding_size)
        self.intersection_mlp = MlpOperator(embedding_size)
        self.negation_mlp = MlpOperatorSingle(embedding_size)


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

        query_scores = self.decoder(query_encoding)
        return query_scores

    
    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels).type(torch.LongTensor)
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

        return self.projection_mlp(sub_query_encoding, relation_embeddings)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        
        return  reduce(lambda x, y: self.intersection_mlp(x, y), sub_query_encoding_list)

    
    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        

        return reduce(lambda x, y: self.union_mlp(x, y), sub_query_encoding_list)


    def negation(self, sub_query_encoding):
        """
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        return self.negation_mlp(sub_query_encoding)



class MLPMixerReasoner(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, 
                 dropout_rate=0.3,
             value_vocab=None, ):
        super(MLPMixerReasoner, self).__init__(num_entities, num_relations, embedding_size)


        self.num_entities = num_entities
        self.num_relations = num_relations

    
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
       
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

    
        self.value_vocab = value_vocab

      

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()



        

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

     
        
        
        # MLP operations

        self.projection_mlp = MLPMixer(embedding_size)
        self.union_mlp = MLPMixer(embedding_size)
        self.intersection_mlp = MLPMixer(embedding_size)
        self.negation_mlp = MlpOperatorSingle(embedding_size)


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

        query_scores = self.decoder(query_encoding)
        return query_scores

    
    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels).type(torch.LongTensor)
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

        return self.projection_mlp(sub_query_encoding, relation_embeddings)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        
        return  reduce(lambda x, y: self.intersection_mlp(x, y), sub_query_encoding_list)

    
    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        

        return reduce(lambda x, y: self.union_mlp(x, y), sub_query_encoding_list)


    def negation(self, sub_query_encoding):
        """
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        return self.negation_mlp(sub_query_encoding)



def test_mlp():
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

    gqe_model = MLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()

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


def test_mlp_mixer():
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

    gqe_model = MLPMixerReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()

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


if __name__ == "__main__":
    test_mlp()
    test_mlp_mixer()