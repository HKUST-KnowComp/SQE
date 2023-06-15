import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Parameter
from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss
from hype_util import *



class OffsetSet(nn.Module):
    """
    Layer for setting Offsets for box/vec model.
    """
    def __init__(self, manifold, mode_dims, expand_dims, agg_func=torch.min, name='Real_offset'):
        super(OffsetSet, self).__init__()
        
        self.manifold = manifold
        #default true for offset_use_center
        self.agg_func = agg_func

        self.pre_mats = ManifoldParameter(torch.FloatTensor(expand_dims*2, mode_dims), manifold=self.manifold)
        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s"%name, self.pre_mats)

        self.post_mats = ManifoldParameter(torch.FloatTensor(mode_dims, expand_dims), manifold=self.manifold)
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):

        temp1 = torch.cat([embeds1, embeds1_o], dim=1)
        temp2 = torch.cat([embeds2, embeds2_o], dim=1)
        if len(embeds3_o) > 0:
            temp3 = torch.cat([embeds3, embeds3_o], dim=1)
    
        temp1 = F.relu(temp1.mm(self.pre_mats))
        temp2 = F.relu(temp2.mm(self.pre_mats))
        if len(embeds3_o) > 0:
            temp3 = F.relu(temp3.mm(self.pre_mats))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class InductiveOffsetSet(nn.Module):
    """
    Layer for Inductive offset version.
    """
    def __init__(self, manifold, mode_dims, expand_dims, off_reg, agg_func=torch.min, name='Real_offset'):
        super(InductiveOffsetSet, self).__init__()
        self.manifold = manifold
        #default true for offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(self.manifold, mode_dims, expand_dims, self.agg_func)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3_o) > 0:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o, embeds3_o]), dim=0)[0]
        else:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o]), dim=0)[0]
        offset = offset_min * F.sigmoid(self.OffsetSet_Module( embeds1, embeds1_o, embeds2, embeds2_o, embeds3, embeds3_o))
        return offset

class Attention(nn.Module):
    """
    Class for calculation attention (supports AttentionSet)
    Attributes
    """
    def __init__(self, manifold, mode_dims, expand_dims, name="Real"):
        super(Attention, self).__init__()
        
        self.manifold = manifold
        #default ele for att_type
        #default no for bn
        #default 1 for nat
        #default true for center_use_offset
        self.atten_mats1 = ManifoldParameter(torch.FloatTensor(expand_dims*2, mode_dims), manifold=self.manifold)
        
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter("atten_mats1_%s"%name, self.atten_mats1)

        self.atten_mats2 = ManifoldParameter(torch.FloatTensor(mode_dims, mode_dims), manifold=self.manifold)
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter("atten_mats2_%s"%name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        
        temp1 = torch.cat([center_embed, offset_embed], dim=1)
        temp2 = F.relu(temp1.mm(self.atten_mats1))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3

class AttentionSet(nn.Module):
    """
    Layer for Attention aggregated Centers
    """
    def __init__(self, manifold, mode_dims, expand_dims, att_reg=0., att_tem=1., name="Real"):
        super(AttentionSet, self).__init__()
        #default true for center_use_offset
        self.manifold = manifold
        self.att_reg = att_reg
        self.att_tem = att_tem
        self.Attention_module = Attention(self.manifold, mode_dims, expand_dims)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg)/(self.att_tem+1e-4)
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg)/(self.att_tem+1e-4)
        #default ele
        if len(embeds3) > 0:
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg)/(self.att_tem+1e-4)
            combined = F.softmax(torch.stack([temp1, temp2, temp3]), dim=0)
            center = embeds1*combined[0] + embeds2*combined[1] + embeds3*combined[2]
        else:
            combined = F.softmax(torch.stack([temp1, temp2]), dim=0)
            center = embeds1*combined[0] + embeds2*combined[1]

        return center



#MAIN MODEL
class HypE(IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, alpha=0.02, gamma = 24, curvature = 1.0):
        super(HypE, self).__init__(num_entities, num_relations, embedding_size)

        #Hyperparameters set-up
        self.embedding_size = embedding_size
        self.c = nn.Parameter(torch.FloatTensor([curvature]),requires_grad=False)
        self.manifold = PoincareBall(self.c)
        self.func = F.relu

        self.alpha = nn.Parameter( torch.Tensor([alpha]),requires_grad=False)
        self.epsilon = 2.0
        self.gamma_num = gamma
        self.gamma = ManifoldParameter(torch.Tensor([gamma]), requires_grad=False,manifold = self.manifold)

        self.embedding_range = ManifoldParameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / embedding_size]), 
            requires_grad=False,
            manifold = self.manifold
        )
        #Entity and Relation Embeddings
        self.entity_embedding = ManifoldParameter(torch.zeros(num_entities, self.embedding_size), manifold=self.manifold)
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        
        self.relation_embedding = ManifoldParameter(torch.zeros(num_relations, self.embedding_size), manifold=self.manifold)
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.offset_embedding = ManifoldParameter(torch.zeros(num_relations, self.embedding_size), manifold=self.manifold)
        nn.init.uniform_(tensor=self.offset_embedding, a=0., b=self.embedding_range.item())

        #center and offset network use final combination stated from the paper: eleattention & inductive deepset
        self.center_sets = AttentionSet(self.manifold, self.embedding_size, self.embedding_size, 
                                                    att_reg = 0., att_tem=1.)
        self.offset_sets = InductiveOffsetSet(self.manifold, self.embedding_size, self.embedding_size, off_reg = 0., agg_func=torch.mean)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)


    def scoring(self, query_hype_embedding):


        entity_embeddings = self.entity_embedding
        query_center_embedding, query_offset_embedding = query_hype_embedding

        # [1, num_entities, embedding_size]
        enlarged_entity_embeddings = entity_embeddings.unsqueeze(0)

        # [batch_size, 1, embedding_size]
        enlarged_center_embeddings = query_center_embedding.unsqueeze(1)

        # [batch_size, 1, embedding_size]
        enlarged_offset_embeddings = query_offset_embedding.unsqueeze(1)

        query_min = enlarged_center_embeddings - 0.5*enlarged_offset_embeddings
        query_max = enlarged_center_embeddings + 0.5*enlarged_offset_embeddings

        dist_out = (F.relu(enlarged_entity_embeddings - query_max) + F.relu(query_min - enlarged_entity_embeddings)).sum(dim=-1)

        dist_in = (enlarged_center_embeddings - torch.minimum( query_max, torch.maximum(query_min, enlarged_entity_embeddings))).abs().sum(dim=-1)

        distances = dist_out + self.alpha * dist_in
        return self.gamma_num - distances

    def projection(self, relation_ids, sub_query_hype_embedding):
        """
        The relational projection of hype

        :param relation_ids: [batch_size]
        :param sub_query_center_embedding: [batch_size, embedding_size]
        :param sub_query_offset_embedding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size], [batch_size, embedding_size] (center + offset)
        """
        # print(len(sub_query_hype_embedding))
        sub_query_center_embedding, sub_query_offset_embedding = sub_query_hype_embedding
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(
            self.relation_embedding.device)

        relation_embeddings = self.relation_embedding[relation_ids.long()]
        offset_embeddings = self.offset_embedding[relation_ids.long()]
        new_center_embedding = relation_embeddings + sub_query_center_embedding
        new_offset_embedding = offset_embeddings + sub_query_offset_embedding
        """
        query_center = head
            for rel in range(rel_len):
                query_center = query_center + relation[:,rel,:,:]
        """
        #TODO: Apply special addition here for manifolds
        new_hype_embedding = tuple([new_center_embedding, new_offset_embedding])

        return new_hype_embedding
    

    def higher_projection(self, relation_ids, sub_query_hype_embedding):
        return self.projection(relation_ids, sub_query_hype_embedding)

    def intersection(self, sub_query_hype_embedding_list):
        """
        :param: sub_query_hype_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
        :return:  [batch_size, embedding_size], [batch_size, embedding_size]
        """
        sub_query_center_embedding_list, sub_query_offset_embedding_list = sub_query_hype_embedding_list


        new_query_center_embeddings = self.center_sets(sub_query_center_embedding_list[0].squeeze(1), sub_query_offset_embedding_list[0].squeeze(1), 
                                                        sub_query_center_embedding_list[1].squeeze(1), sub_query_offset_embedding_list[1].squeeze(1))
        new_query_offset_embeddings = self.offset_sets(sub_query_center_embedding_list[0].squeeze(1), sub_query_offset_embedding_list[0].squeeze(1),
                                                        sub_query_center_embedding_list[1].squeeze(1), sub_query_offset_embedding_list[1].squeeze(1))

        new_query_hype_embeddings = tuple([new_query_center_embeddings, new_query_offset_embeddings])

        return new_query_hype_embeddings


    def loss_fnt(self, sub_query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(sub_query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss


    def forward(self, batched_structured_query, label=None):
        # We need to override this forward function as the structure of hype embedding is different
        # input: batched_structured_query
        # output: HYPE TUPLE instead of single embedding
        assert batched_structured_query[0] in ["p", "e", "i"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            # intersection of hype embedding takes tuple of two lists
            sub_query_center_result_list = []
            sub_query_offset_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_center_result, sub_query_offset_result = self.forward(batched_structured_query[_i])
                sub_query_center_result_list.append(sub_query_center_result)
                sub_query_offset_result_list.append(sub_query_offset_result)

            sub_query_hype_result_list = tuple([sub_query_center_result_list, sub_query_offset_result_list])
            this_query_result = self.intersection(sub_query_hype_result_list)

        elif batched_structured_query[0] == "e":
            # set the offset tensor to all zeros
            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.device)
            #print(entity_ids.type())
            this_query_center_result = self.entity_embedding[entity_ids.long()]
            this_query_offset_result = torch.zeros(this_query_center_result.shape).to(
                self.entity_embedding.device)
            this_query_result = tuple([this_query_center_result, this_query_offset_result])

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

    hype_model = HypE(num_entities=nentity, num_relations=nrelation, embedding_size=300)
    if torch.cuda.is_available():
        hype_model = hype_model.cuda()

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

        query_embedding = hype_model(batched_query)
        print("center dimension:", query_embedding[0].shape)
        print("offset dimension:", query_embedding[1].shape)
        print(len(positive_sample))

        loss = hype_model(batched_query, positive_sample)
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

            query_embedding = hype_model(batched_query)
            result_logs = hype_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = hype_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
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

            query_embedding = hype_model(batched_query)
            result_logs = hype_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = hype_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
