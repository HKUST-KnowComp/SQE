import argparse
from gqe import GQE
from q2b import Q2B
from q2p import Q2P
from betae import BetaE
from cone import ConE
from transformer import TransformerModel
from biqe import BiQEModel
from rnn import RNNModel
from gru import GRUModel
from lstm import LSTMModel
from tree_lstm import TreeLSTM
from tree_rnn import TreeRNN
from tcn import TCNModel
from hype import HypE
from hype_util import RiemannianAdam
from mlp import MLPMixerReasoner, MLPReasoner
from fuzzqe import FuzzQE

import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import json


def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work. 
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True
            
        
        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument("--train_query_dir", required=True)
    parser.add_argument("--valid_query_dir", required=True)
    parser.add_argument("--test_query_dir", required=True)
    parser.add_argument('--kg_data_dir', default="KG_data/", help="The path the original kg data")

    parser.add_argument('--num_layers', default = 3, type=int, help="num of layers for sequential models")

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--checkpoint_path", type=str, default="../logs")
    parser.add_argument("-old", "--old_loss_fnt", action="store_true")
    parser.add_argument("-fol", "--use_full_fol", action="store_true")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--few_shot", type=int, default=32)

    args = parser.parse_args()

    KG_data_path = "../" + args.kg_data_dir
    data_name = args.data_name
    
  
    train_query_file_names = []
    valid_query_file_names = []
    test_query_file_names = []

    for file in os.listdir(args.train_query_dir):
        if file.endswith(".json") and data_name in file:
            train_query_file_names.append(file)

    for file in os.listdir(args.valid_query_dir):
        if file.endswith(".json") and data_name in file:
            valid_query_file_names.append(file)
    
    for file in os.listdir(args.test_query_dir):
        if file.endswith(".json") and data_name in file:
            test_query_file_names.append(file)

    
    if args.few_shot != 32:
        train_query_file_names = train_query_file_names[:args.few_shot]
        
    data_path = KG_data_path + args.data_name
    fol_type = None
    fol_type = "pinu" if args.use_full_fol else "pi"
    print("fol_type: ", fol_type)
    if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
        fol_type = "pi"
    loss = "old-loss" if args.old_loss_fnt else "new-loss"
    #new_loss: label smoothing for iterative models, default for sequential models
    #old_loss: negative sampling for iterative models
    info = fol_type + "_" + loss + "_" + str(args.few_shot)

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.checkpoint_path + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name[:-6] + '/train'
    test_log_dir = args.checkpoint_path  + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name[:-6] + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    batch_size = args.batch_size

    evaluating_query_types = []
    evaluating_df = pd.read_csv("../preprocess/test_generated_formula_anchor_node=3_filtered.csv").reset_index(drop=True)

    for i in range(evaluating_df.shape[0]):
        query_structure = evaluating_df.original[i]
        evaluating_query_types.append(query_structure)

    print("Evaluating query types: ", evaluating_query_types)

    training_query_types = []
    training_df = pd.read_csv("../preprocess/test_generated_formula_anchor_node=2.csv").reset_index(drop=True)

    for i in range(training_df.shape[0]):
        query_structure = training_df.original[i]
        training_query_types.append(query_structure)
    print("Training query types: ", training_query_types)

   

    # create model
    print("====== Initialize Model ======", args.model)
    if args.model == 'gqe':
        model = GQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, use_old_loss=args.old_loss_fnt)
    elif args.model == 'q2b':
        model = Q2B(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, use_old_loss=args.old_loss_fnt)
    elif args.model == "q2p":
        model = Q2P(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "transformer":
        model = TransformerModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim ,num_layers = args.num_layers)
    elif args.model == "biqe":
        model = BiQEModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim ,num_layers = args.num_layers)
    elif args.model == "rnn":
        model = RNNModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers)
    elif args.model == "gru":
        model = GRUModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers)
    elif args.model == "lstm":
        model = LSTMModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers)
    elif args.model == "betae":
        model = BetaE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "cone":
        model = ConE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, use_old_loss=args.old_loss_fnt) 
    elif args.model == "tcn":
        model = TCNModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "hype":
        model = HypE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "tree_lstm":
        model = TreeLSTM(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "tree_rnn":
        model = TreeRNN(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "mlp":
        model = MLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "mlp_mixer":
        model = MLPMixerReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "fuzzqe":
        model = FuzzQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    else:
        raise NotImplementedError

    # add scheduler for the transformer model to do warmup, or the model will not converge at all
    optimizer = None
    if args.model == "hype":
         optimizer = RiemannianAdam(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=args.learning_rate
                    )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )

    if args.model == "transformer" or args.model == "biqe":

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        def warmup_lambda(epoch):
            if epoch < args.warm_up_steps:
                return epoch * 1.0 / args.warm_up_steps
            else:
                return 1


        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    if args.model == "betae" or args.model == "cone":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )

        def warmup_lambda(epoch):
            if epoch < args.warm_up_steps:
                return epoch * 1.0 / args.warm_up_steps
            else:
                return 1

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if torch.cuda.is_available():
        model = model.cuda()



    
    global_steps = -1

    file_count = -1
    model_name = args.model
    while True:
    # for train_query_file_name in train_query_file_names:
        print(train_query_file_names)
        file_count += 1
        train_query_file_name = np.random.choice(train_query_file_names)

        print("====== Training ======", model_name, train_query_file_name)

        with open(args.train_query_dir + "/" + train_query_file_name , "r") as fin:
            train_data_dict = json.load(fin)


        train_iterators = {}
        for query_type, query_answer_dict in train_data_dict.items():
                #for these three model, only process conjunctive queries
                if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
                    if "u" in query_type or "n" in query_type:
                        continue
                #if the non-fol parameter is passed (sequential), also use conjunctive queries
                if args.use_full_fol == False:
                    if "u" in query_type or "n" in query_type:
                        continue

               
                
    
                new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDataset(nentity, nrelation, query_answer_dict),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=TrainDataset.collate_fn
                ))
                train_iterators[query_type] = new_iterator

        train_iteration_names = list(train_iterators.keys())
        
        total_length = 0
        for key, value in train_iterators.items():
            total_length += value.len

        total_step = total_length * 20

        # Train the model
        for step in tqdm(range(total_step)):
            global_steps += 1

            model.train()
            
            
            task_name = np.random.choice(train_iteration_names)
            iterator = train_iterators[task_name]
            batched_query, unified_ids, positive_sample = next(iterator)
            
            if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe":
                batched_query = unified_ids
            
            loss = model(batched_query, positive_sample)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (global_steps + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            else: 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            if args.model == "transformer" or args.model == "betae" or args.model == "cone" or args.model == "biqe":
                scheduler.step()
            
            if global_steps % 200 == 0:
                train_summary_writer.add_scalar("y-train-" + task_name, loss.item(), global_steps)
            
            save_step = args.log_steps
            model_name = args.model

           
            # Evaluate the model
            if global_steps % args.log_steps == 0:

                # Save the model
                model.eval()
                general_checkpoint_path = args.checkpoint_path + "/" + model_name + "_" + str(global_steps) + "_" + info + "_" + data_name + ".bin"

                
                torch.save({
                    'steps': global_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            

                # Validation
                print("====== Validation ======", model_name)
                entailment_58_types_logs = []
                entailment_29_types_logs = []
                entailment_unseen_29_types_logs = []

                generalization_58_types_logs = []
                generalization_29_types_logs = []
                generalization_unseen_29_types_logs = []

                entailment_58_types_dict = {}
                generalization_58_types_dict = {}
        
                for valid_query_file_name in tqdm(valid_query_file_names):

                    with open(args.valid_query_dir + "/" + valid_query_file_name , "r") as fin:
                        valid_data_dict = json.load(fin)
                    

                    validation_loaders = {}
                    for query_type, query_answer_dict in valid_data_dict.items():

                        if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
                            if "u" in query_type or "n" in query_type:
                                continue

                        if args.use_full_fol == False:
                            if "u" in query_type or "n" in query_type:
                                continue

                        new_iterator = DataLoader(
                            ValidDataset(nentity, nrelation, query_answer_dict),
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=ValidDataset.collate_fn
                        )
                        validation_loaders[query_type] = new_iterator

                    

                    for task_name, loader in validation_loaders.items():

                        all_entailment_logs = []
                        all_generalization_logs = []

                        for batched_query, unified_ids, train_answers, valid_answers in loader:

                            if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe":
                                batched_query = unified_ids

                            query_embedding = model(batched_query)
                            entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                            generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

                            all_entailment_logs.extend(entailment_logs)
                            all_generalization_logs.extend(generalization_logs)

                            if task_name in evaluating_query_types:
                                entailment_58_types_logs.extend(entailment_logs)
                                generalization_58_types_logs.extend(generalization_logs)

                            if task_name in training_query_types:
                                entailment_29_types_logs.extend(entailment_logs)
                                generalization_29_types_logs.extend(generalization_logs)

                            if task_name in evaluating_query_types and task_name not in training_query_types:
                                entailment_unseen_29_types_logs.extend(entailment_logs)
                                generalization_unseen_29_types_logs.extend(generalization_logs)

                        
                        if task_name not in entailment_58_types_dict:
                            entailment_58_types_dict[task_name] = []
                        entailment_58_types_dict[task_name].extend(all_entailment_logs)


                        if task_name not in generalization_58_types_dict:
                            generalization_58_types_dict[task_name] = []
                        generalization_58_types_dict[task_name].extend(all_generalization_logs)

                            
                for task_name, logs in entailment_58_types_dict.items():
                    aggregated_entailment_logs = log_aggregation(logs)
                    for key, value in aggregated_entailment_logs.items():
                        test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, global_steps)
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, global_steps)
                
                
                entailment_58_types_logs = log_aggregation(entailment_58_types_logs)
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
                entailment_29_types_logs = log_aggregation(entailment_29_types_logs)
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
                entailment_unseen_29_types_logs = log_aggregation(entailment_unseen_29_types_logs)
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                

                for key, value in entailment_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-29-types-" + key, value, global_steps)

                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-29-types-" + key, value, global_steps)

                for key, value in entailment_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-unseen-29-types-" + key, value, global_steps)

                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-unseen-29-types-" + key, value, global_steps)


                for key, _ in entailment_58_types_logs.items():
                    macro_average = (entailment_29_types_logs[key] + entailment_unseen_29_types_logs[key]) / 2
                    test_summary_writer.add_scalar("x-valid-58-types-" + key, macro_average, global_steps)

                for key, _ in generalization_58_types_logs.items():
                    macro_average = (generalization_29_types_logs[key] + generalization_unseen_29_types_logs[key]) / 2
                    test_summary_writer.add_scalar("x-valid-58-types-" + key, macro_average, global_steps)

                
                print("====== Test ======", model_name)
                entailment_58_types_logs = []
                entailment_29_types_logs = []
                entailment_unseen_29_types_logs = []

                generalization_58_types_logs = []
                generalization_29_types_logs = []
                generalization_unseen_29_types_logs = []

                entailment_58_types_dict = {}
                generalization_58_types_dict = {}

                for test_query_file_name in tqdm(test_query_file_names):
                    with open(args.test_query_dir + "/" + test_query_file_name , "r") as fin:
                        test_data_dict = json.load(fin)
                    

                    test_loaders = {}
                    for query_type, query_answer_dict in test_data_dict.items():

                        if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
                            if "u" in query_type or "n" in query_type:
                                continue

                        if args.use_full_fol == False:
                            if "u" in query_type or "n" in query_type:
                                continue

                        new_iterator = DataLoader(
                            TestDataset(nentity, nrelation, query_answer_dict),
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=TestDataset.collate_fn
                        )
                        test_loaders[query_type] = new_iterator

                    

                    for task_name, loader in test_loaders.items():

                        all_entailment_logs = []
                        all_generalization_logs = []

                        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:

                            if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe":
                                batched_query = unified_ids

                            query_embedding = model(batched_query)
                            entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                            generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                            all_entailment_logs.extend(entailment_logs)
                            all_generalization_logs.extend(generalization_logs)

                            if task_name in evaluating_query_types:
                                entailment_58_types_logs.extend(entailment_logs)
                                generalization_58_types_logs.extend(generalization_logs)

                            if task_name in training_query_types:
                                entailment_29_types_logs.extend(entailment_logs)
                                generalization_29_types_logs.extend(generalization_logs)

                            if task_name in evaluating_query_types and task_name not in training_query_types:
                                entailment_unseen_29_types_logs.extend(entailment_logs)
                                generalization_unseen_29_types_logs.extend(generalization_logs)

                        if task_name not in entailment_58_types_dict:
                            entailment_58_types_dict[task_name] = []
                        entailment_58_types_dict[task_name].extend(all_entailment_logs)


                        if task_name not in generalization_58_types_dict:
                            generalization_58_types_dict[task_name] = []
                        generalization_58_types_dict[task_name].extend(all_generalization_logs)
                    
                
                for task_name, logs in entailment_58_types_dict.items():
                    aggregated_entailment_logs = log_aggregation(logs)
                    for key, value in aggregated_entailment_logs.items():
                        test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, global_steps)
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, global_steps)
                

                
                entailment_58_types_logs = log_aggregation(entailment_58_types_logs)
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
                entailment_29_types_logs = log_aggregation(entailment_29_types_logs)
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
                entailment_unseen_29_types_logs = log_aggregation(entailment_unseen_29_types_logs)
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                for key, _ in entailment_58_types_logs.items():
                    macro_average = (entailment_29_types_logs[key] + entailment_unseen_29_types_logs[key]) / 2
                    test_summary_writer.add_scalar("x-test-58-types-" + key, macro_average, global_steps)

                for key, _ in generalization_58_types_logs.items():
                    macro_average = (generalization_29_types_logs[key] + generalization_unseen_29_types_logs[key]) / 2
                    test_summary_writer.add_scalar("x-test-58-types-" + key, macro_average, global_steps)

                for key, value in entailment_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

                for key, value in entailment_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)

                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)

        
                gc.collect()
        

















