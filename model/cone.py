import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss


pi = 3.14159265358979323846


def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)  
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings

class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding


class ConE(IterativeModel):
    """
    In cone embedding, each entities and queries are encoded with a d-dimensional cone embedding (theta_ax, theta_ap)
    It's very close to Q2B, only the geometric property of cones makes it possible for negation and union.
    The implementation of iterative operators are different.
    """
    def __init__(self, num_entities, num_relations, alpha=150, gamma = 12, embedding_size = 800, label_smoothing=0.1, center_reg=0.02, drop=0., negative_size = 128, use_old_loss = False):
        super(ConE, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size #hyper-parameter 
        self.epsilon = 2.0
        self.cen = center_reg

        self.alpha = alpha
        self.gamma = nn.Parameter(#Not Used For New Loss Function
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / embedding_size]),
            requires_grad=False
        )

        #embeddings for entities
        self.entity_embedding = nn.Embedding(num_entities, self.embedding_size) #axis embedding [num_entities, embedding_size](arg = 0)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        
        
        
        
        
        #self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
        #   nentity).to(torch.float).repeat(test_batch_size, 1) 

        
        self.angle_scale = AngleScale(self.embedding_range.item())
        self.axis_scale = 1.0
        self.arg_scale = 1.0

        # self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)
        #set of axis embeddings [num_entities, embedding_size]
        self.axis_embedding = nn.Embedding(num_entities, self.embedding_size)
        nn.init.uniform_(
            tensor=self.axis_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #set of arg embeddings [num_entities, embedding_size]
        self.arg_embedding = nn.Embedding(num_entities, self.embedding_size)
        nn.init.uniform_(
            tensor=self.arg_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )



        self.cone_proj = ConeProjection(self.embedding_size, 800, 2)
        self.cone_intersection = ConeIntersection(self.embedding_size, drop)
        self.cone_negation = ConeNegation()

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.negative_size = negative_size
        self.use_old_loss = use_old_loss

    
    def distance_cone(self,entity_embedding, query_cone_embedding):
        """
        return distance of entities and queries
        :query_cone_embedding: tuple[query_axis_embedding, query_arg_embedding]
        for each embedding: [batch_size * embedding_dim]
        """
        #this function takes original implementation from ConE repo
        query_axis_embedding, query_arg_embedding = query_cone_embedding
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out,p=1,dim=-1) + self.cen * torch.norm(distance_in,p=1,dim=-1)

        # print("distance:", distance, distance.shape)
        return distance

    
    
    def scoring(self, query_cone_embedding):
        

        #:query_cone_embedding: tuple[query_axis_embedding, query_arg_embedding]
        #:return: [batch_size, num_entities]
        
        entity_embeddings = self.entity_embedding.weight
        query_axis_embedding, query_arg_embedding = query_cone_embedding
        # [1, num_entities, embedding_size]
        enlarged_entity_embeddings = entity_embeddings.unsqueeze(0)
        # [batch_size, 1, embedding_size]
        enlarged_axis_embeddings = query_axis_embedding.unsqueeze(1)
        # [batch_size, 1, embedding_size]
        enlarged_arg_embeddings = query_arg_embedding.unsqueeze(1)

        enlarged_cone_embeddings = tuple([enlarged_axis_embeddings,enlarged_arg_embeddings])
        
        distances = self.distance_cone(enlarged_entity_embeddings,enlarged_cone_embeddings)
        #print(distances.size())
        #print(torch.mean(distances,dim = -1))

    
        
        scores = self.alpha - distances
        #print(scores[0])
        # print("scores:", scores, scores.shape)
        return scores

    def old_loss_fnt(self, query_cone_encoding, labels):

        # The size of the query_cone_encoding is ([batch_size, embedding_size],[batch_size, embedding_size]) 

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
        query_axis_embedding, query_arg_embedding = query_cone_encoding

        # [batch_size, embedding_size]
        positive_distance = self.distance_cone(label_entity_embeddings, query_cone_encoding)
        negative_distance = self.distance_cone(negative_samples_embeddings, query_cone_encoding)

        # Original Repo Normalized logits inside logsigmoid
        margin = self.alpha
        positive_score = -F.logsigmoid(margin - positive_distance)
        # print(positive_score.shape)
        negative_score = -F.logsigmoid(negative_distance - margin)  # .mean(dim=1)
        # Currently the negative sample size is 1, so no need for mean
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
    

    def projection(self, relation_ids, sub_query_cone_embedding):
        """
        The relational projection of ConE. 

        :param relation_ids: [batch_size]
        :param sub_query_cone_embedding: tuple([batch_size, embedding_size],[batch_size, embedding_size])
        :return: tuple([batch_size, embedding_size],[batch_size, embedding_size])
        """
        #embedding from the previous query
        axis_embedding, arg_embedding = sub_query_cone_embedding
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(
            self.axis_embedding.weight.device)
        #index_select relations from model parameter
        relation_axis_embeddings = self.axis_embedding(relation_ids)
        relation_arg_embeddings = self.arg_embedding(relation_ids)

        relation_axis_embeddings = self.angle_scale(relation_axis_embeddings, self.axis_scale)
        relation_arg_embeddings = self.angle_scale(relation_arg_embeddings, self.arg_scale)

        relation_axis_embeddings = convert_to_axis(relation_axis_embeddings)
        relation_arg_embeddings = convert_to_axis(relation_arg_embeddings)

        new_axis_embedding, new_arg_embedding = self.cone_proj(axis_embedding, arg_embedding, relation_axis_embeddings, relation_arg_embeddings)
        query_cone_embedding = tuple([new_axis_embedding, new_arg_embedding])

        return query_cone_embedding

    def higher_projection(self, relation_ids, sub_query_cone_embedding):
        return self.projection(relation_ids, sub_query_cone_embedding)

    def union(self, sub_query_cone_embedding_list):
    
    #:param: sub_query_cone_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
    #:return:  tuple([batch_size, embedding_size], [batch_size, embedding_size])
    # we use de morgan's law to compute the union of the cone embeddings:  A or B =  not ( ( not A ) and ( not B ))
        axis_embedding_list, arg_embedding_list = sub_query_cone_embedding_list
        #negation1
        negated_axis_list = []
        negated_arg_list = []
        for i in range(len(axis_embedding_list)):
            axis_embedding, arg_embedding = self.cone_negation(axis_embedding_list[i], arg_embedding_list[i])
            negated_axis_list.append(axis_embedding)
            negated_arg_list.append(arg_embedding)
        #intersection
        negated_embedding_list = tuple([negated_axis_list, negated_arg_list])
        intersected_axis_list, intersected_arg_list = self.intersection(negated_embedding_list)
        #negation2
        final_axis_embedding, final_arg_embedding = self.cone_negation(intersected_axis_list, intersected_arg_list)

        union_query_embedding = tuple([final_axis_embedding, final_arg_embedding])
        return union_query_embedding
    
    def intersection(self, sub_query_cone_embedding_list):
        """
        :param: sub_query_cone_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
        :return:  tuple([batch_size, embedding_size], [batch_size, embedding_size])
        """
        axis_embedding_list, arg_embedding_list = sub_query_cone_embedding_list
        stacked_axis_embeddings = torch.stack(axis_embedding_list)
        stacked_arg_embeddings = torch.stack(arg_embedding_list)

        axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)
        all_subquery_encodings = tuple([axis_embedding, arg_embedding])
        return all_subquery_encodings
    
    def negation(self, query_cone_embedding):
        axis_embedding, arg_embedding = query_cone_embedding
        axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)
        negation_query_encoding = axis_embedding, arg_embedding
        return negation_query_encoding

    
    def forward(self, batched_structured_query, label=None):

        assert batched_structured_query[0] in ["p", "e", "i", "n", "u"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            
            sub_query_axis_result_list = []
            sub_query_arg_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_axis_result, sub_query_arg_result = self.forward(batched_structured_query[_i])
                sub_query_axis_result_list.append(sub_query_axis_result)
                sub_query_arg_result_list.append(sub_query_arg_result)

            sub_query_cone_result_list = tuple([sub_query_axis_result_list, sub_query_arg_result_list])
            this_query_result = self.intersection(sub_query_cone_result_list)

        elif batched_structured_query[0] == "u":

            sub_query_axis_result_list = []
            sub_query_arg_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_axis_result, sub_query_arg_result = self.forward(batched_structured_query[_i])
                sub_query_axis_result_list.append(sub_query_axis_result)
                sub_query_arg_result_list.append(sub_query_arg_result)

            sub_query_cone_result_list = tuple([sub_query_axis_result_list, sub_query_arg_result_list])
            this_query_result = self.union(sub_query_cone_result_list)

        elif batched_structured_query[0] == "n":
            sub_query_result = self.forward(batched_structured_query[1])
            this_query_result = self.negation(sub_query_result)

        elif batched_structured_query[0] == "e":
            
            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            axis_entity_embedding = self.entity_embedding(entity_ids)
            axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
            axis_entity_embedding = convert_to_axis(axis_entity_embedding)
            arg_entity_embedding = torch.zeros_like(axis_entity_embedding).to(
                self.entity_embedding.weight.device)

            this_query_result = tuple([axis_entity_embedding,arg_entity_embedding])

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            if self.use_old_loss:
                loss = self.old_loss_fnt(this_query_result, label)                
            else:
                loss = self.loss_fnt(this_query_result, label)
            return loss
  

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

    cone_model =ConE(num_entities=nentity, num_relations=nrelation, embedding_size=300,use_old_loss=False)
    #if torch.cuda.is_available():
        #cone_model = cone_model.cuda()

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():

        if "u" in query_type:
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

        query_embedding = cone_model(batched_query)
        print(query_embedding[0].shape)
        print(query_embedding[1].shape)
        loss = cone_model(batched_query, positive_sample)
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

            query_embedding = cone_model(batched_query)
            result_logs = cone_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = cone_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

        if "u" in query_type:
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

            query_embedding = cone_model(batched_query)
            result_logs = cone_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = cone_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)


            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
