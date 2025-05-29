#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MODEL_NAME_OR_PATH="/data/Qwen2.5-0.5B-Instruct" # model path

EVAL_DATASETS="/data/align_anything_t2t" # rm dataset path
EVAL_TEMPLATE="HOMEWORK" # dataset template
EVAL_SPLIT="validation" # split the dataset

OUTPUT_DIR="../outputs/qwen_2_5_rm_base" # output dir

# For wandb online logging
export WANDB_API_KEY="8bac9db5cb3498e52d6f1def5f8beaba39fbccbd"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 1 \
     --epochs 1