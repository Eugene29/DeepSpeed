# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist

seq_shard_size_list = None

def set_seq_shard_size_list(lst):
    global seq_shard_size_list
    seq_shard_size_list = lst

def get_seq_shard_size_list():
    global seq_shard_size_list
    return seq_shard_size_list