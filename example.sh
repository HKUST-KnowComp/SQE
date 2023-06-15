########################################################################
####################### Example Training Scripts #######################
########################################################################

#Training GQE on FB15k-237

CUDA_VISIBLE_DEVICES=0 python ../model/train.py \
    -dn FB15k-237-betae \
    -m gqe \
    --train_query_dir /DIR/sampled_data_29_train \
    --valid_query_dir /DIR/sampled_data_58_valid \
    --test_query_dir /DIR/sampled_data_58_test \
    --checkpoint_path /DIR/kg_reasoning_logs \
    -b 8192 

#Training Q2P on FB15k, add "fol" to train model on First-order logic queries

CUDA_VISIBLE_DEVICES=0 python ../model/train.py \
  -dn FB15k-betae \
  -m q2p \
  --train_query_dir /DIR/sampled_data_29_train \
  --valid_query_dir /DIR/sampled_data_58_valid \
  --test_query_dir /DIR/sampled_data_58_test \
  --checkpoint_path /DIR/kg_reasoning_logs \
  -fol  \
  -b 1024

#Training BetaE on FB15k, we use gradient accumulation to maintain the batch-size for BetaE and ConE.

CUDA_VISIBLE_DEVICES=1 python ../model/train.py \
    -dn FB15k-betae \
    -m betae \
    --train_query_dir /DIR/sampled_data_29_train \
    --valid_query_dir /DIR/sampled_data_58_valid \
    --test_query_dir /DIR/sampled_data_58_test \
    --checkpoint_path /DIR/kg_reasoning_logs \
    -b 32 \
    --log_steps 60000 \
    --gradient_accumulation_steps 32 \
    --warm_up_steps 10000 \
    -fol \
    -lr 0.0003

#Training SQE-LSTM on FB15k-237

CUDA_VISIBLE_DEVICES=0 python ../model/train.py \
    -dn FB15k-237-betae \
    -m lstm \
    --train_query_dir /DIR/sampled_data_29_train \
    --valid_query_dir /DIR/sampled_data_58_valid \
    --test_query_dir /DIR/sampled_data_58_test \
    --checkpoint_path /DIR/kg_reasoning_logs \
    -b 1024 \
    --log_steps 120000 \
    -fol \
    -lr 0.0001

#Training SQE-Transformer on NELL

CUDA_VISIBLE_DEVICES=0 python ../model/train.py \
    -dn NELL-betae \
    -m transformer \
    --train_query_dir /DIR/sampled_data_29_train \
    --valid_query_dir /DIR/sampled_data_58_valid \
    --test_query_dir /DIR/sampled_data_58_test \
    --checkpoint_path /DIR/kg_reasoning_logs \
    -b 512 \
    --log_steps 120000 \
    -fol \
    -lr 0.0001
