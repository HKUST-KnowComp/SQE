import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class BetaE(IterativeModel):
    """
    In beta embedding, we encode each entity/relation/query into vector space of embedding_size. 
    Each alpha and beta embedding takes half of the embedding size.
    splitting:  alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
    concating:  embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
    """
    def __init__(self, num_entities, num_relations, embedding_size = 500, num_layers = 2, gamma = 1, label_smoothing=0.0):
        super(BetaE, self).__init__(num_entities, num_relations, embedding_size)

        self.gamma = nn.Parameter(  # Margin when calculating score
            torch.Tensor([gamma]),
            requires_grad=False
        )

        #embeddings for entity & relation
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + 2.0) / embedding_size]), 
            requires_grad=False
        )
        self.entity_embedding = nn.Embedding(num_entities, embedding_size * 2)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
        self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        self.embedding_size = embedding_size #hyper-parameter 
        self.num_layers = num_layers #hyper-parameter, default set to 2 according to origin repository 

        self.center_net = BetaIntersection(embedding_size)
        self.projection_net = BetaProjection(embedding_size * 2, 
                                             self.embedding_size, 
                                             self.embedding_size, 
                                             self.projection_regularizer, 
                                             self.num_layers)

        

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)



    def distance_beta(self,entity_embedding, query_encoding):

        entity_alpha_embedding, entity_beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        query_alpha_embedding, query_beta_embedding = torch.chunk(query_encoding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha_embedding, entity_beta_embedding)
        query_dist = torch.distributions.beta.Beta(query_alpha_embedding, query_beta_embedding)
        
        distance = torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return distance

    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """
        entity_embeddings = self.entity_regularizer(self.entity_embedding.weight)
         # [1, num_entities, embedding_size]
        enlarged_entity_embeddings = entity_embeddings.unsqueeze(0)
        # [batch_size, 1, embedding_size]
        enlarged_query_encoding = query_encoding.unsqueeze(1)
        #print(enlarged_query_encoding.size())
        distances = self.distance_beta(enlarged_entity_embeddings,enlarged_query_encoding)
        scores = 95 - distances
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

    def old_loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, embedding_size]
        # and the labels are [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)
        label_entity_embeddings = self.entity_regularizer(self.entity_embedding(labels))

        batch_size = label_entity_embeddings.shape[0]
        #print(batch_size)
        # [batch_size]
        random_entity_indices = torch.randint(0, self.num_entities, (batch_size,)).to(
            self.entity_embedding.weight.device)

        # [batch_size, embedding_size]
        negative_samples_embeddings = self.entity_regularizer(self.entity_embedding(random_entity_indices))
        
        # [batch_size, embedding_size]
        positive_distance = self.distance_beta(label_entity_embeddings, query_encoding)
        negative_distance = self.distance_beta(negative_samples_embeddings, query_encoding)

        # Original Repo Normalized logits inside logsigmoid
        margin = self.gamma
        positive_score = -F.logsigmoid(margin - positive_distance)
        negative_score = -F.logsigmoid(negative_distance - margin)  # .mean(dim=1)
        # Currently the negative sample size is 1, so no need for mean
        relu = nn.ReLU()
        loss = torch.mean(relu(positive_score + negative_score))
        return loss

    def projection(self, relation_ids, sub_query_encoding):
        """
        The relational projection of BetaE. 

        :param relation_ids: [batch_size]
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        relation_embeddings = self.relation_embedding(relation_ids)

        query_embedding = self.projection_net(sub_query_encoding, relation_embeddings)

        return query_embedding

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)

    def union(self, sub_query_encoding_list):
        """
        :param sub_query_encoding_list: [[batch_size, embedding_size], [batch_size, embedding_size], ...]
        :return: [batch_size, embedding_size]
        De Morgan's law: U(a1b1,a2b2,...) = N[ N(a1,b1) intersect N(a2,b2) intersect ...]
        """

        #a for loop with length = num_operands   
        negated_embedding_list = [self.negation(sub_query_encoding) for sub_query_encoding in sub_query_encoding_list]
        intersect_query_embedding = self.intersection(negated_embedding_list)
        union_query_embedding = self.negation(intersect_query_embedding)
        return union_query_embedding

    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=0)
        all_alpha_embedding, all_beta_embedding = torch.chunk(all_subquery_encodings, 2, dim=-1)
        alpha_embedding, beta_embedding = self.center_net(all_alpha_embedding, all_beta_embedding)
        all_subquery_encodings = torch.cat([alpha_embedding, beta_embedding], dim=-1)
        return all_subquery_encodings
    
    def negation(self, query_encoding):
        #N(alpha, beta) = (1/alpha, 1/beta)
        negation_query_encoding =  1./query_encoding
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
            this_query_result = self.entity_regularizer(raw_entity_embedding)

        else:
            this_query_result = None
        if label is None:
            
            return this_query_result

        else:

            return self.loss_fnt(this_query_result, label)

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

    betae_model =BetaE(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    #if torch.cuda.is_available():
        #betae_model = betae_model.cuda()

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

        query_embedding = betae_model(batched_query)
        print(query_embedding.shape)
        loss = betae_model(batched_query, positive_sample)
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

            query_embedding = betae_model(batched_query)
            result_logs = betae_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = betae_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
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

            query_embedding = betae_model(batched_query)
            result_logs = betae_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = betae_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)


            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
