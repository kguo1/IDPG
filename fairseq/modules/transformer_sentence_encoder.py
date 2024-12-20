# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import numpy as np
from fairseq.data import encoders

import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from fairseq import utils
from fairseq.modules import (
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
import random



def kron(A, B):
    return torch.einsum('ij,kl->ikjl', A, B).reshape(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)

def init_adapter_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.000001)
        if module.bias is not None:
            module.bias.data.zero_()

class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        dictionary, 
        bpe,
        sentence_generation,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        add_prefix: bool = False,
        prefix_len: int = 0,
        prefix_prompt: torch.Tensor = None,
        add_suffix: bool = False,
        suffix_len: int = 5,
        suffix_prompt: torch.Tensor = None,
        generation_dropout: float = 0.1, 
        generation_activation_fn: str = "gelu",
        generation_net: str = "dnn",
        generation_layer: int = 1,
        insert_position: int = 4,
        sbert: bool = False,
        generation_quaternions: int = None,
        lphm: int = None,
        middle_prompt_mode: str = 'none',
        middle_prompt_insert_layer: int = 25, 
        middle_previous: bool = False,
        adapter_arch: str = 'none',
        adapter_insert_layer: int = 25,
        adapter_bottleneck_dim: int = 64,
        compacter_n: int = 4,
        generator_layer_norm: bool = False,
        generator_residual: bool = False,
        reparameterization: str = 'None', 
        phm_bottleneck_dim: int = 16,
        prompt_insert_mode: int = 1, 
        glove_path: str = None,
        custom_insert_position_fraction: float = 0,
        original_max_positions: int = 256,
        args = None,
        #prefix_prompt: torch.Tensor = torch.tensor([713, 16, 10, 205, 1569]),
    ) -> None:




        super().__init__()
        self.dictionary = dictionary
        self.bpe = bpe

        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.generation_quaternions = generation_quaternions
        self.lphm = lphm
        self.reparameterization = reparameterization
        self.phm_bottleneck_dim = phm_bottleneck_dim
        self.prompt_insert_mode = prompt_insert_mode
        self.custom_insert_position_fraction = custom_insert_position_fraction
        self.original_max_positions = original_max_positions
        self.args = args


        if reparameterization != 'None': 
            self.re_lstm_head = torch.nn.LSTM(input_size=self.embedding_dim,
                hidden_size=self.embedding_dim // 2,
                num_layers=2,
                dropout=0,
                bidirectional=True,
                batch_first=True)
            self.re_mlp_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim))

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        # self
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )


        print("max sequence length", max_seq_len)
        print(original_max_positions, 'original max positions')

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_transformer_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=self.dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size
            )
            for _ in range(num_encoder_layers)
        ])

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            # actually probably want to unfreeze this if insert position is size
            if insert_position != 6:
                freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        self.add_prefix = add_prefix
        self.prefix_len = prefix_len
        self.prefix_prompt = prefix_prompt
        self.add_suffix = add_suffix
        self.suffix_len = suffix_len
        self.suffix_prompt = suffix_prompt
        self.sentence_generation = sentence_generation

        if self.sentence_generation == 'glove':
            self.word2idx = {}
            with open(glove_path, 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    self.word2idx[word] = np.array(line[1:]).astype(np.float)
            self.generator_dim = int(glove_path.split('.')[-2][:-1])
        elif sentence_generation == 'roberta':
            self.generator_dim = embedding_dim

        self.insert_position = insert_position
        self.generation_net = generation_net
        self.generation_layer = generation_layer
        #self.mid_dim = mid_dim
        #self.prefix_MLP_mode = prefix_MLP_mode
        #assert self.prefix_MLP_mode in ['none', 'shared', 'separate']
        #self.prefix_pos = prefix_pos
        self.middle_prompt_insert_layer = middle_prompt_insert_layer
        self.middle_prompt_mode = middle_prompt_mode
        self.adapter_insert_layer = adapter_insert_layer
        self.adapter_arch = adapter_arch
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.compacter_n = compacter_n
        self.generator_residual = generator_residual
        self.residual_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.multi_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.middle_previous = middle_previous
        if generator_layer_norm:
            self.generator_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.generator_layer_norm = False


        self.count = 0

        assert self.middle_prompt_insert_layer in [i for i in range(1, 26)]
        assert self.middle_prompt_mode in ['none', 'shared', 'layerb']
        assert self.adapter_arch in ['none', 'houlsby', 'compacter', 'pfeiffer']

        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.uniform_(-0.5, 0.5)
                #nn.init.uniform_(module, a=-0.5, b=0.5)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        if self.adapter_arch == 'houlsby':
            self.middle_adapter_MLP = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(self.embedding_dim, self.adapter_bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_bottleneck_dim, self.embedding_dim)) for _ in range(25 - self.adapter_insert_layer)])

            self.middle_adapter_MLP2 = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(self.embedding_dim, self.adapter_bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_bottleneck_dim, self.embedding_dim)) for _ in range(25 - self.adapter_insert_layer)])

            self.middle_adapter_MLP.apply(init_adapter_params)
            self.middle_adapter_MLP2.apply(init_adapter_params)

        if self.adapter_arch == 'compacter':
            self.compacter_down_proj_s = ParameterList([Parameter(torch.empty(self.embedding_dim // compacter_n, 1, requires_grad=True)) for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2)])
            self.compacter_down_proj_t = ParameterList([Parameter(torch.empty(1, self.adapter_bottleneck_dim // compacter_n, requires_grad=True)) for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2)])
            self.compacter_down_proj_b = ParameterList([Parameter(torch.empty(self.adapter_bottleneck_dim, requires_grad=True)) for i in range((25 - self.adapter_insert_layer) * 2)])

            self.compacter_up_proj_s = ParameterList([Parameter(torch.empty(self.adapter_bottleneck_dim // compacter_n, 1, requires_grad=True)) for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2)])
            self.compacter_up_proj_t = ParameterList([Parameter(torch.empty(1, self.embedding_dim // compacter_n, requires_grad=True)) for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2)])
            self.compacter_up_proj_b = nn.Embedding((25 - self.adapter_insert_layer) * 2, self.embedding_dim)

            self.compacter_shared_A = ParameterList([Parameter(torch.empty(compacter_n, compacter_n, requires_grad=True)) for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2)])

            for i in range(compacter_n * (25 - self.adapter_insert_layer) * 2):
                nn.init.normal_(self.compacter_down_proj_s[i], mean=0.0, std=0.05)
                nn.init.normal_(self.compacter_down_proj_t[i], mean=0.0, std=0.05)
                nn.init.normal_(self.compacter_up_proj_s[i], mean=0.0, std=0.05)
                nn.init.normal_(self.compacter_up_proj_t[i], mean=0.0, std=0.05)
                nn.init.normal_(self.compacter_shared_A[i], mean=0.0, std=0.05)

            for i in range((25 - self.adapter_insert_layer) * 2):
                nn.init.normal_(self.compacter_down_proj_b[i], mean=0.0, std=0.05)

            self.compacter_up_proj_b.apply(_init_weights)

        if self.adapter_arch == 'pfeiffer':
            self.middle_adapter_MLP = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(self.embedding_dim, self.adapter_bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_bottleneck_dim, self.embedding_dim)) for _ in range(25 - self.adapter_insert_layer)])
            self.middle_adapter_MLP.apply(init_adapter_params)

        if self.add_suffix:
            if self.sentence_generation is not None:
                self.test = True
                if generation_net == 'dnn':
                    if generation_layer == 1:
                        self.generation_dense = nn.Linear(self.generator_dim, self.suffix_len * self.embedding_dim)
                        self.generation_dense.apply(init_bert_params)
                    else:
                        if generation_quaternions is not None:
                            if lphm is None:

                                # the B matrices and the bias
                                self.generation_dense_w = ParameterList([Parameter(torch.empty(self.generator_dim // generation_quaternions, self.phm_bottleneck_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_dense_b = Parameter(torch.empty(self.phm_bottleneck_dim, requires_grad=True))

                                # The B matrices and the bias
                                # print("self suffix length", self.suffix_len)
                                self.generation_out_proj_w = ParameterList([Parameter(torch.empty(self.phm_bottleneck_dim // generation_quaternions, self.prompt_insert_mode * self.suffix_len * self.embedding_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_out_proj_b = nn.Embedding(self.suffix_len * self.prompt_insert_mode, self.embedding_dim)

                                # A in the phm setup
                                self.generation_shared_w1 = ParameterList([Parameter(torch.empty(generation_quaternions, generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_shared_w2 = ParameterList([Parameter(torch.empty(generation_quaternions, generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])

                                for i in range(generation_quaternions):
                                    nn.init.normal_(self.generation_dense_w[i], mean=0.0, std=0.05)
                                    nn.init.normal_(self.generation_out_proj_w[i], mean=0.0, std=0.05)
                                    nn.init.normal_(self.generation_shared_w1[i], mean=0.0, std=0.05)
                                    nn.init.normal_(self.generation_shared_w2[i], mean=0.0, std=0.05)
                                    #nn.init.kaiming_uniform_(self.generation_dense_w[i], a=math.sqrt(5))
                                    #nn.init.kaiming_uniform_(self.generation_out_proj_w[i], a=math.sqrt(5))
                                    #nn.init.kaiming_uniform_(self.generation_shared_w[i], a=math.sqrt(5))
                                    #nn.init.zeros_(self.generation_dense_w[i])
                                    #nn.init.zeros_(self.generation_out_proj_w[i])
                                    #nn.init.zeros_(self.generation_shared_w1[i])
                                    #nn.init.zeros_(self.generation_shared_w2[i])


                                nn.init.normal_(self.generation_dense_b, mean=0.0, std=0.05)
                                #nn.init.zeros_(self.generation_dense_b)
                                #nn.init.uniform_(self.generation_out_proj_b, a=-0.04, b=0.04)

                                self.generation_out_proj_b.apply(_init_weights)
                            else:
                                self.generation_dense_s = ParameterList([Parameter(torch.empty(self.generator_dim // generation_quaternions, lphm, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_dense_t = ParameterList([Parameter(torch.empty(1, self.phm_bottleneck_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_dense_b = Parameter(torch.empty(self.phm_bottleneck_dim, requires_grad=True))

                                self.generation_out_proj_s = ParameterList([Parameter(torch.empty(self.phm_bottleneck_dim // generation_quaternions, lphm, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_out_proj_t = ParameterList([Parameter(torch.empty(lphm, self.suffix_len * self.embedding_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])
                                self.generation_out_proj_b = nn.Embedding(self.suffix_len, self.embedding_dim)

                                self.generation_shared_w = ParameterList([Parameter(torch.empty(generation_quaternions, generation_quaternions, requires_grad=True)) for i in range(generation_quaternions)])

                                for i in range(generation_quaternions):
                                    nn.init.normal_(self.generation_dense_s[i], mean=0.0, std=0.2236)
                                    nn.init.normal_(self.generation_dense_t[i], mean=0.0, std=0.2236)
                                    nn.init.normal_(self.generation_out_proj_s[i], mean=0.0, std=0.2236)
                                    nn.init.normal_(self.generation_out_proj_t[i], mean=0.0, std=0.2236)
                                    nn.init.normal_(self.generation_shared_w[i], mean=0.0, std=0.05)

                                    #nn.init.zeros_(self.generation_dense_s[i])
                                    #nn.init.zeros_(self.generation_dense_t[i])
                                    #nn.init.zeros_(self.generation_out_proj_s[i])
                                    #nn.init.zeros_(self.generation_out_proj_t[i])
                                    #nn.init.zeros_(self.generation_shared_w[i])

                                nn.init.normal_(self.generation_dense_b, mean=0.0, std=0.05)
                                #nn.init.zeros_(self.generation_dense_b)
                                self.generation_out_proj_b.apply(_init_weights)
                        else:
                            self.generation_dense = nn.Linear(self.generator_dim, self.phm_bottleneck_dim)
                            if self.middle_prompt_insert_layer == 25:
                                self.generation_out_proj = nn.Linear(self.phm_bottleneck_dim, self.suffix_len * self.embedding_dim)
                            else:
                                self.generation_out_proj = nn.Linear(self.phm_bottleneck_dim, self.suffix_len * self.embedding_dim, bias=False)


                            self.generation_dense.apply(init_bert_params)
                            self.generation_out_proj.apply(init_bert_params)
                    self.generation_activation_fn = utils.get_activation_fn(generation_activation_fn)
                    self.generation_dropout = nn.Dropout(p=generation_dropout)

                elif generation_net == 'rnn':
                    self.generation_lstm = nn.LSTM(
                        input_size=self.generator_dim,
                        hidden_size=self.embedding_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True,
                    )
                    self.generation_dense_after_lstm = nn.Linear(2 * self.embedding_dim, self.phm_bottleneck_dim)
                    self.generation_activation_fn = utils.get_activation_fn(generation_activation_fn)
                    self.generation_dropout = nn.Dropout(p=generation_dropout)
                    self.generation_out_proj = nn.Linear(self.phm_bottleneck_dim, self.suffix_len * self.embedding_dim)

                    self.generation_lstm.apply(init_bert_params)
                    self.generation_dense_after_lstm.apply(init_bert_params)
                    self.generation_out_proj.apply(init_bert_params)

                if self.middle_prompt_insert_layer < 25:
                    # minus one is because last year's cls is not affected by the inserted prompt
                    # with no need to insert in last layer
                    self.middle_prompt_insert_num = 25 - self.middle_prompt_insert_layer - 1

                    if self.middle_prompt_mode == 'none':
                        if self.generation_quaternions  is not None:
                            self.middle_prompt = nn.Embedding(self.suffix_len * self.prompt_insert_mode * self.middle_prompt_insert_num, self.embedding_dim)
                            self.middle_prompt.apply(init_bert_params)
                            print(type(self.middle_prompt))
                            print(self.middle_prompt(torch.arange(self.suffix_len)))

                            self.middle_generation_dense_w = ParameterList([Parameter(torch.empty(self.generator_dim // generation_quaternions, self.phm_bottleneck_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions * self.middle_prompt_insert_num)])
                            self.middle_generation_dense_b = ParameterList([Parameter(torch.empty(self.phm_bottleneck_dim, requires_grad=True)) for i in range(self.middle_prompt_insert_num)])


                            self.middle_generation_out_proj_w = ParameterList([Parameter(torch.empty(self.phm_bottleneck_dim // generation_quaternions, self.prompt_insert_mode * self.suffix_len * self.embedding_dim // generation_quaternions, requires_grad=True)) for i in range(generation_quaternions * self.middle_prompt_insert_num)])
                            self.middle_generation_out_proj_b = nn.Embedding(self.prompt_insert_mode * self.suffix_len * self.middle_prompt_insert_num, self.embedding_dim)

                            self.middle_generation_shared_w1 = ParameterList([Parameter(torch.empty(generation_quaternions, generation_quaternions, requires_grad=True)) for i in range(generation_quaternions * self.middle_prompt_insert_num)])
                            self.middle_generation_shared_w2 = ParameterList([Parameter(torch.empty(generation_quaternions, generation_quaternions, requires_grad=True)) for i in range(generation_quaternions * self.middle_prompt_insert_num)])

                            for i in range(generation_quaternions * self.middle_prompt_insert_num):
                                nn.init.normal_(self.middle_generation_dense_w[i], mean=0.0, std=0.05)
                                nn.init.normal_(self.middle_generation_out_proj_w[i], mean=0.0, std=0.05)
                                nn.init.normal_(self.middle_generation_shared_w1[i], mean=0.0, std=0.05)
                                nn.init.normal_(self.middle_generation_shared_w2[i], mean=0.0, std=0.05)

                            for i in range(self.middle_prompt_insert_num):
                                nn.init.normal_(self.middle_generation_dense_b[i], mean=0.0, std=0.05)

                            self.middle_generation_out_proj_b.apply(_init_weights)
                        else:
                            self.middle_generation_dense = nn.ModuleList([nn.Linear(in_features=self.generator_dim, out_features=self.phm_bottleneck_dim) for i in range(self.middle_prompt_insert_num)])
                            self.middle_generation_out_proj = nn.ModuleList([nn.Linear(self.phm_bottleneck_dim, self.suffix_len * self.embedding_dim) for i in range(self.middle_prompt_insert_num)])
                            for i in range(self.middle_prompt_insert_num):
                                self.middle_generation_dense[i].apply(init_bert_params)
                                self.middle_generation_out_proj[i].apply(init_bert_params)

                    elif self.middle_prompt_mode == 'layerb':
                        if self.generation_quaternions  is not None:
                            self.middle_generation_out_proj_b = nn.Embedding(self.prompt_insert_mode * self.suffix_len * self.middle_prompt_insert_num, self.embedding_dim)
                            self.middle_generation_out_proj_b.apply(_init_weights)
                        else:
                            self.middle_prompt_insert_num += 1
                            self.middle_generation_out_proj_b = nn.Embedding(self.prompt_insert_mode * self.suffix_len * self.middle_prompt_insert_num, self.embedding_dim)
                            self.middle_generation_out_proj_b.apply(_init_weights)
            else:
                if self.suffix_prompt == None:
                    self.suffix_embed = nn.Embedding(self.suffix_len * self.prompt_insert_mode, self.embedding_dim)
                    self.suffix_embed.apply(_init_weights)

                    #self.middle_prompt = nn.Embedding(5 * 24, self.embedding_dim)
                    #self.middle_prompt.apply(_init_weights)
                else:
                    self.suffix_embed = nn.Embedding.from_pretrained(self.embed_tokens(self.suffix_prompt), freeze=False)

                if self.middle_prompt_insert_layer < 25 and self.middle_prompt_mode == 'none':
                    self.middle_prompt_insert_num = 25 - self.middle_prompt_insert_layer
                    if self.middle_prompt_mode == 'none':
                        self.middle_prompt = nn.Embedding(self.suffix_len * self.prompt_insert_mode * self.middle_prompt_insert_num, self.embedding_dim)
                        self.middle_prompt.apply(_init_weights)
                        #print(self.middle_prompt(torch.arange(self.suffix_len, self.suffix_len * 2)))
                    elif self.middle_prompt_mode == 'adapter-shared':
                        self.middle_adapter_MLP = nn.Sequential(
                            nn.Linear(self.embedding_dim, self.adapter_bottleneck_dim),
                            nn.Tanh(),
                            nn.Linear(self.adapter_bottleneck_dim, self.embedding_dim))
                        self.middle_adapter_MLP.apply(init_bert_params)
                    else:
                        self.middle_adapter_MLP = nn.ModuleList(
                            [nn.Sequential(
                                nn.Linear(self.embedding_dim, self.adapter_bottleneck_dim),
                                nn.GELU(),
                                nn.Linear(self.adapter_bottleneck_dim, self.embedding_dim)) for _ in range(self.middle_prompt_insert_num)])
                        self.middle_adapter_MLP.apply(init_bert_params)

        
        #if self.apply_bert_init:
        #    self.apply(init_bert_params)
    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        ans = ''
        for s in sentences:
            if s[-1] == self.dictionary.pad():
                for idx in range(len(s)):
                    if s[idx] == self.dictionary.pad():
                        break
                s = s[:idx]
            ans += ' ' + self.bpe.decode(self.dictionary.string(s))
        return ans

    def glove_encode(self, tokens: str):
        count = 0
        weights_matrix = np.zeros((self.generator_dim))
        for s in tokens.split():
            words_stripped = ''.join(c for c in s if not c in string.punctuation)
            word = words_stripped.lower()
            if word in self.word2idx:
                weights_matrix += self.word2idx[word]
                count += 1
        if count != 0:
            return weights_matrix / count
        else:
            return np.random.normal(scale=0.6, size=(self.generator_dim, ))

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
    ):
        return TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def prompt_forward(
        self,
        x: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # add [CLS], [SEP] into generated prompt
        # T x 5 x C -> T x 7 x C
        new_x = torch.cat([self.embed_tokens(torch.LongTensor([0]).to(f"cuda:{self.args.device_id}")).repeat(x.shape[0], 1).type_as(x).view(x.shape[0], -1, x.shape[-1]), x, self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")).repeat(x.shape[0], 1).type_as(x).view(x.shape[0], -1, x.shape[-1])], dim=1)

        # compute padding mask. This is needed for multi-head attention
        padding_mask = None

        if self.embed_positions is not None:
            zeros = torch.zeros(x.shape[0], self.suffix_len + 2).to(f"cuda:{self.args.device_id}")
            new_x += self.embed_positions(zeros, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            # TODO: if segment_labels is not None needs to shift
            new_x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            new_x = self.quant_noise(new_x)

        if self.emb_layer_norm is not None:
            new_x = self.emb_layer_norm(new_x)

        new_x = F.dropout(new_x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            new_x *= 1 - padding_mask.unsqueeze(-1).type_as(new_x)

        # B x T x C -> T x B x C
        new_x = new_x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(new_x)

        for idx, layer in enumerate(self.layers):
            new_x, _ = layer(new_x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(new_x)

        sentence_rep = new_x[0, :, :]

        if last_state_only:
            inner_states = [new_x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if tokens.size()[1] > 292:
            print("more than half the original max tokens size")
            print('tokens size', tokens.size())
            raise MemoryError
        print('tokens size', tokens.size())
        # print('token', tokens)
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if self.sentence_generation is not None:
            if self.sentence_generation == 'roberta':
                x = self.embed_tokens(tokens) # embed the tokens

                if self.embed_scale is not None:
                    x *= self.embed_scale # scale embeddings

                if self.embed_positions is not None:
                    x += self.embed_positions(tokens, positions=positions)
                    # add positional embeddings

                if self.segment_embeddings is not None and segment_labels is not None:
                    x += self.segment_embeddings(segment_labels)

                if self.quant_noise is not None:
                    x = self.quant_noise(x)

                if self.emb_layer_norm is not None:
                    x = self.emb_layer_norm(x)

                x = F.dropout(x, p=self.dropout, training=self.training)

                # account for padding while computing the representation
                if padding_mask is not None:
                    x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

                for layer in self.layers:
                    x, _ = layer(x, self_attn_padding_mask=padding_mask)
                    # x = x.to('cpu').
                    # x = x.to(f"cuda:{self.args.device_id}")

                inner_states = [x]
                torch.cuda.empty_cache()
            else:
                x = np.array([])
                x = np.zeros((tokens.size()[0], self.generator_dim), dtype='float32' if self.generation_shared_w1[0][0].dtype==torch.float32 else 'half')
                for i in range(tokens.size()[0]):
                    x[i] += self.glove_encode(self.decode(tokens[i]))
                x = torch.from_numpy(x).to(f"cuda:{self.args.device_id}")
                inner_states = [x]

        x = self.embed_tokens(tokens)
        x = x.to('cpu').detach()
        # print('moved x to cpu')
        # print("we have done forward pass once")

        torch.cuda.empty_cache()

        if self.add_suffix:
            if self.sentence_generation is not None:
                if self.generation_net == 'dnn':
                    if self.sentence_generation == 'roberta':
                        features = inner_states[-1].transpose(0, 1)[:, 0, :]
                        # get the first token for all of the batches
                    else:
                        features = inner_states[-1]

                    if not features.is_cuda:
                        features = features.to(f"cuda:{self.args.device_id}")

                    suffix_x = self.generation_dropout(features)
                    # suffix_x = suffix_x.to("cpu")
                    #suffix x is size B x C

                    if self.middle_prompt_insert_layer < 24 and not self.middle_previous:
                        suffix_x0 = suffix_x.clone() # we dont seem to enter here

                    if self.generator_residual:

                        # repeats 5 times in the second dimension, so becomes like b x (c * 5)
                        # then reshape to B x 5 x C
                        residual_x = suffix_x.repeat(1, 5).view(suffix_x.size()[0], 5, self.generator_dim)

                    if self.generation_quaternions is not None:
                        
                        # dont put tmp_x on device
                        tmp_x = self.generation_dense_b.repeat(suffix_x.size()[0], 1).to(f"cuda:{self.args.device_id}")
                        
                        # tmp_x = self.generation_dense_b.repeat(suffix_x.size()[0], 1)
                        # if tmp_x.is_cuda:
                        #     tmp_x = tmp_x.to('cpu')
                        
                        
                        # for each batch, store the bias weights
                        # shape B x m
                        
                        for i in range(self.generation_quaternions):
                            # for i in n (phm setup in paper)

                            # suffix_x = suffix_x.to(f"cuda:{self.args.device_id}")
                            if self.lphm is None:

                                # this basically passes x through the first layer of phm
                                tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w1[i], self.generation_dense_w[i]).to(f"cuda:{self.args.device_id}"))
                                #tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w1[i], self.generation_dense_w[i]).to(f"cuda:{self.args.device_id}")).to('cpu')
                            else: 
                                tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w[i], torch.mm(self.generation_dense_s[i], self.generation_dense_t[i])).to(f"cuda:{self.args.device_id}"))
                            # suffix_x = suffix_x.to('cpu')
                        
                        torch.cuda.empty_cache()
                        # suffix_x = suffix_x.to(f"cuda:{self.args.device_id}")
                        print(suffix_x.device, "suffix x device")
                        suffix_x = self.generation_activation_fn(tmp_x)
                        suffix_x = self.generation_dropout(suffix_x)
                        # should be shape b x m 
                        # activation and dropout

                        # grab the bias for each token from embedding matrix
                        # shape t x d. view(-1) flattens to a 1d of length t x d
                        # then repeats for batch size so for each batch we have t x d vector 
                        # thus shape is B x (t * d)
                        # tmp_x = self.generation_out_proj_b(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).view(-1).repeat(suffix_x.size()[0], 1).to(f"cuda:{self.args.device_id}")
                        tmp_x = self.generation_out_proj_b(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).view(-1).repeat(suffix_x.size()[0], 1)
                        # if tmp_x.is_cuda:
                        #     tmp_x = tmp_x.to('cpu')

                        for i in range(self.generation_quaternions):
                            if self.lphm is None:
                                tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w2[i], self.generation_out_proj_w[i]).to(f"cuda:{self.args.device_id}"))
                                #tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w2[i], self.generation_out_proj_w[i]).to(f"cuda:{self.args.device_id}")).to("cpu")
                            else:
                                tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w[i], torch.mm(self.generation_out_proj_s[i], self.generation_out_proj_t[i])).to(f"cuda:{self.args.device_id}"))


                    
                        # pass through the second layer of phm 
                        # should be shape B x (t * d)
                        # now we can then probably reshape to be B x t x d. I understand it now
                        suffix_x = tmp_x
                        print("suffix device", suffix_x.device)
                        if suffix_x.is_cuda:
                            suffix_x = suffix_x.to('cpu').detach()
                        torch.cuda.empty_cache()
                        # 16 x 5120
                    else:
                        suffix_x = self.generation_dense(suffix_x)
                        suffix_x = self.generation_activation_fn(suffix_x)
                        suffix_x = self.generation_dropout(suffix_x)
                        if self.generation_layer == 2:
                            suffix_x = self.generation_out_proj(suffix_x)
                            if self.middle_prompt_insert_layer != 25:
                                suffix_x += self.middle_generation_out_proj_b(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).view(-1).repeat(suffix_x.size()[0], 1).to(f"cuda:{self.args.device_id}")

                    # reshape to B x suffix_len x d
                    if self.generator_residual:
                        suffix_x = residual_x + self.residual_layer_norm(suffix_x.view(tokens.size()[0], -1, self.embedding_dim))
                    else:  # we enter here
                        suffix_x = suffix_x.view(tokens.size()[0], -1, self.embedding_dim)

                    if self.generator_layer_norm != False:
                        suffix_x = self.generator_layer_norm(suffix_x)
                elif self.generation_net == 'rnn':
                    features = inner_states[-1].transpose(0, 1)[:, 1:, :]
                    suffix_x, _ = self.generation_lstm(features)
                    suffix_x = torch.cat(
                        [
                            suffix_x[:, :, :self.embedding_dim],
                            torch.flip(suffix_x[:, :, self.embedding_dim:], [1]),
                        ],
                        -1,
                    )
                    suffix_x = suffix_x[:, 0, :]
                    suffix_x = self.generation_dropout(suffix_x)
                    suffix_x = self.generation_dense_after_lstm(suffix_x)
                    suffix_x = self.generation_activation_fn(suffix_x)
                    suffix_x = self.generation_dropout(suffix_x)
                    suffix_x = self.generation_out_proj(suffix_x)
                    suffix_x = suffix_x.view(tokens.size()[0], -1, self.embedding_dim)
                if self.generation_net == 'dnn1':
                    suffix_x = self.suffix_embed(torch.arange(5).type_as(tokens))
            else:
                if self.reparameterization != 'None':
                    input_embeds = self.suffix_embed(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).unsqueeze(0)
                    self.output_embeds = self.re_mlp_head(self.re_lstm_head(input_embeds)[0]).squeeze()
                else:
                    suffix_x = self.suffix_embed(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).repeat(tokens.size()[0], 1).view(-1, self.suffix_len * self.prompt_insert_mode, self.embedding_dim)
            
            # print("we have finished creating the suffix")
            torch.cuda.empty_cache()
            # we eval to true 
            if self.prompt_insert_mode == 1:
                if x.is_cuda:
                    x = x.to("cpu")
                new_x = []
                new_padding_mask = []
                self.stop_idxs = []  # very important
                missing_padding_lengths = []
                # print('suffix_x shape')
                # print(suffix_x.shape)


                # if self.insert_position == 6:
                #     # find the last stop token

                #     largest_last_sep_index = 0
                #     for i in range(tokens.size()[0]):
                #         for last_sep_index in range(tokens.size()[1] - 1, tmp, -1):
                #             if tokens[i][last_sep_index] == 2:
                #                 largest_last_sep_index = max(largest_last_sep_index, last_sep_index)
                #                 break
                        

                # this should be iterating through the batch because
                # tokens size should be B x T
                length_of_suffix_added = 0
                if self.insert_position == 6:
                    actual_suffix_length = self.suffix_len // (self.original_max_positions)


                    # print("updating and saving length of suffix added")
                    length_of_suffix_added = (tokens.size()[1]) * actual_suffix_length 
                # print('length of suffix added', length_of_suffix_added)
                
                # print("length of suffix added", length_of_suffix_added)
                # print(length_of_suffix_added_tmp, "length of suffix added temp")
                # print("largest last sep index", largest_last_sep_index)
                for i in range(tokens.size()[0]):

                    # iterate through sequence length, break once we find the first separator token
                    for stop_idx in range(tokens.size()[1]):
                        if tokens[i][stop_idx] == 2:
                            break
                    if self.insert_position == 0:
                        if self.reparameterization != 'None':
                            new_x.append(torch.cat((x[i][:1], self.output_embeds, x[i][1:]), 0))
                        else: # we enter
                            # we add the batch suffix in desired position 
                            new_x.append(torch.cat((x[i][:1], self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], x[i][1:]), 0))

                        stop_idx = 1
                    elif self.insert_position == 5:
                        # print(self.insert_position, "insert position is five, custom insertion")
                        # we will specify a place in the input sequence with a fraction
                        insertion_index_fraction = int(self.custom_insert_position_fraction * x[i].shape[0])
                        # print(insertion_index_fraction, "actual insertion index")
                        if self.reparameterization != 'None':
                            new_x.append(torch.cat((x[i][:insertion_index_fraction], self.output_embeds, x[i][insertion_index_fraction:]), 0))
                        else:
                            new_x.append(torch.cat((x[i][:insertion_index_fraction], self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], x[i][insertion_index_fraction:]), 0))

                        stop_idx = insertion_index_fraction
                    elif self.insert_position == 6:

                        tmp = stop_idx

                        # length_of_suffix_added_tmp =  0

                        largest_last_sep_index = tokens.size()[1] - 1



                      

                        #find the last stop token
                        for last_sep_index in range(tokens.size()[1] - 1, -1, -1):
                            if tokens[i][last_sep_index] == 2:
                                break
                        
                        
                        
                        # we need to insert at every position, and we need stop indexes for every position
                        input_stop_idxs = []
                        intermediate_x = None
                        for j in range(last_sep_index):
                            if intermediate_x is None:
                                intermediate_x = torch.cat((x[i][j:j + 1], suffix_x[i][j * actual_suffix_length: (j + 1) * actual_suffix_length]), 0 )
                            else:
                                intermediate_x = torch.cat((intermediate_x, x[i][j:j + 1], suffix_x[i][j * actual_suffix_length: (j + 1) * actual_suffix_length]), 0 )

                            # if last_sep_index == largest_last_sep_index:
                            #     length_of_suffix_added_tmp += actual_suffix_length
                            input_stop_idxs.append(j + 1)
                        
                        # intermediate_x = torch.cat((intermediate_x, x[i][last_sep_index: last_sep_index + 1], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")), suffix_x[i][largest_last_sep_index * actual_suffix_length: (largest_last_sep_index + 1) * actual_suffix_length],  self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")), x[i][last_sep_index+1:]   ))
                        intermediate_x = torch.cat((intermediate_x, x[i][last_sep_index: last_sep_index + 1], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")).to("cpu"), suffix_x[i][largest_last_sep_index * actual_suffix_length: (largest_last_sep_index + 1) * actual_suffix_length],  self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")).to("cpu"), x[i][last_sep_index+1:]   ))
                        input_stop_idxs.append(last_sep_index + 2)
                        stop_idx = input_stop_idxs

                        if last_sep_index == largest_last_sep_index:
                            # length_of_suffix_added_tmp += actual_suffix_length
                            missing_padding_lengths.append(0)
                        else:  # otherwise, we need to pad with ones
                            # calculate the length of the longest input in this batch. 

                            new_longest_input_length = (tokens.size()[1] * actual_suffix_length) + 2 + tokens.size()[1]
                            missing_padding_length = new_longest_input_length - intermediate_x.size(0)
                            missing_padding_lengths.append(missing_padding_length)
                            # missing_padding_length_ones = self.embed_tokens(torch.ones((missing_padding_length), dtype = torch.long).to(f"cuda:{self.args.device_id}"))
                            missing_padding_length_ones = self.embed_tokens(torch.ones((missing_padding_length), dtype = torch.long).to(f"cuda:{self.args.device_id}")).to("cpu")
                            intermediate_x = torch.cat((intermediate_x, missing_padding_length_ones))

                        torch.cuda.empty_cache()


                        # new_x.append(torch.cat((x[i][:stop_idx+1], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")), self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")), x[i][stop_idx+1:]), 0))
                        # stop_idx = stop_idx + 2

                        # assert intermediate_x.size()[0] == (actual_suffix_length * x.size()[1]) + x.size()[1]
                        # print("intermediate x device", intermediate_x.device)
                        new_x.append(intermediate_x)




                    elif self.insert_position == 1:
                        new_x.append(torch.cat((x[i][:stop_idx], self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], x[i][stop_idx:]), 0))
                    elif self.insert_position == 2:
                        new_x.append(torch.cat((x[i][:stop_idx+2], self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], x[i][stop_idx+2:]), 0))
                        stop_idx = stop_idx + 2
                    elif self.insert_position == 3:
                        tmp = stop_idx
                        for stop_idx in range(tokens.size()[1] - 1, tmp, -1):
                            if tokens[i][stop_idx] == 2:
                                break
                        new_x.append(torch.cat((x[i][:stop_idx], self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], x[i][stop_idx:]), 0))
                    else:
                        tmp = stop_idx

                        # find the last stop token
                        for stop_idx in range(tokens.size()[1] - 1, tmp, -1):
                            if tokens[i][stop_idx] == 2:
                                break
                        new_x.append(torch.cat((x[i][:stop_idx+1], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")).to('cpu'), self.suffix_embed(torch.arange(self.suffix_len).type_as(tokens)) if self.sentence_generation is None else suffix_x[i], self.embed_tokens(torch.LongTensor([2]).to(f"cuda:{self.args.device_id}")).to('cpu'), x[i][stop_idx+1:]), 0))
                        stop_idx = stop_idx + 2
                    self.stop_idxs.append(stop_idx)

                    #torch.arange(self.suffix_len).type_as(tokens)
                    # print("padding mask", padding_mask)
                    if padding_mask is not None:  # if there is padding tokens in the input
                        # print("we entered the padding mask section")
                        if self.insert_position in [0, 1,2,3,5]:
                            zeros = torch.zeros(self.suffix_len).type_as(padding_mask)

                        elif self.insert_position == 6:
                            zeros = torch.zeros(length_of_suffix_added + 2).type_as(padding_mask)
                        else:
                            zeros = torch.zeros(self.suffix_len + 2).type_as(padding_mask)


                        # print("zeros device", zeros.device)


                        if self.insert_position != 6:
                            new_padding_mask.append(torch.cat((padding_mask[i][:stop_idx], zeros, padding_mask[i][stop_idx:]), dim=-1))
                        else:
                            intermediate_new_padding_mask = None
                            for ix, stop_indexes in enumerate(stop_idx[:-1]):
                                if intermediate_new_padding_mask is None:
                                    intermediate_new_padding_mask = torch.cat((padding_mask[i][:stop_indexes], zeros[ix*actual_suffix_length:actual_suffix_length*(ix + 1)]), dim = -1)
                                else:
                                    intermediate_new_padding_mask = torch.cat((intermediate_new_padding_mask, padding_mask[i][stop_idx[ix -1]:stop_indexes], zeros[ix*actual_suffix_length:actual_suffix_length*(ix + 1)]), dim = -1)
                            
                            ix = len(stop_idx) - 1
                            stop_indexes = stop_idx[-1]
                            intermediate_new_padding_mask = torch.cat((intermediate_new_padding_mask, padding_mask[i][stop_idx[ix -1]:stop_indexes], zeros[ix*actual_suffix_length:actual_suffix_length*(ix + 1) + 2], padding_mask[i][stop_indexes: ]), dim = -1)
                            
                            if actual_suffix_length*(ix + 1) + 2 < zeros.size()[0]:
                                padding_ones = torch.ones((zeros.size()[0] - (actual_suffix_length*(ix + 1) + 2) )).type_as(intermediate_new_padding_mask)
                                # print("length of padding ones", len(padding_ones))
                                # print("length of zeros", len(zeros))
                                # print("length of padding added for suffix for current input", actual_suffix_length*(ix + 1) + 2)
                                intermediate_new_padding_mask = torch.cat((intermediate_new_padding_mask, padding_ones), dim = -1)
                            
                            # print("intermediate padding mask device", intermediate_new_padding_mask.device)
                            new_padding_mask.append(intermediate_new_padding_mask)
                            
                            # print("padding of the longest input", length_of_suffix_added + 2 + padding_mask.size()[-1])

                            # print('padding size of the current input', intermediate_new_padding_mask.size()[-1])
                            assert intermediate_new_padding_mask.size()[-1] == length_of_suffix_added + 2 + padding_mask.size()[-1]
                            # assert True is False

                x = torch.stack(new_x)
                if padding_mask is not None:
                    padding_mask = torch.stack(new_padding_mask)
                torch.cuda.empty_cache()
        else:
            suffix_x = None

        # print("we have finished adding the suffix to the input")

        if self.embed_scale is not None:
            x *= self.embed_scale
        else:
            self.embed_scale = 1.0

        # if the number of tokens exceeds 512 then truncate. probably change this to max_seq_len
        exceed_len = False

        # print(self.max_seq_len, "max sequence length")
        if self.insert_position == 6:
            if x.size()[1] > self.max_seq_len:
                print(x.size())
                x = x[:, :self.max_seq_len]
                print(x.size())
                print(padding_mask.size())
                if padding_mask is not None:
                    padding_mask = padding_mask[:,:self.max_seq_len]
                print(padding_mask.size())

                exceed_len = True

        else:
            if x.size()[1] > 512:
                print(x.size())
                x = x[:, :512]
                print(x.size())
                print(padding_mask.size())
                if padding_mask is not None:
                    padding_mask = padding_mask[:,:512]
                print(padding_mask.size())

                exceed_len = True



        # if tokens.is_cuda:
        #     tokens = tokens.to('cpu') 

        x = x.to(f"cuda:{self.args.device_id}")
        
        
        torch.cuda.empty_cache()      
        # maybe rewrite this, when zeros are added at the start and not at the position of the prompt we add
        if self.embed_positions is not None:
            if self.add_suffix and self.prompt_insert_mode == 1:
                if self.insert_position in [0,1,2,3,5]:
                    zeros = torch.zeros(x.shape[0], self.suffix_len).type_as(tokens).to(f"cuda:{self.args.device_id}")
                    # zeros = torch.zeros(x.shape[0], self.suffix_len).type_as(tokens)
                elif self.insert_position == 6:
                    zeros = torch.zeros(x.shape[0], length_of_suffix_added + 2).to(f"cuda:{self.args.device_id}")
                    # zeros = torch.zeros(x.shape[0], length_of_suffix_added + 2)
                
                else:
                    # zeros = torch.zeros(x.shape[0], self.suffix_len + 2).type_as(tokens)
                    zeros = torch.zeros(x.shape[0], self.suffix_len + 2).type_as(tokens).to(f"cuda:{self.args.device_id}")
                
                
                tokens = torch.cat([zeros, tokens], dim=-1)
                if exceed_len:
                    if self.insert_position == 6:
                        tokens = tokens[:, :self.max_seq_len]
                    else:
                        tokens = tokens[:, :512]  # probably also need to change this to max seq len

            # if not tokens.is_cuda:
            #     tokens = tokens.to(f"cuda:{self.args.device_id}")
            # if not x.is_cuda:
            #     x = x.to(f"cuda:{self.args.device_id}")
            x += self.embed_positions(tokens, positions=positions)
        torch.cuda.empty_cache()
        # print("we have finished adding the embed positions")

        if self.segment_embeddings is not None and segment_labels is not None:
            # TODO: if segment_labels is not None needs to shift
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # x = x.to("cpu")

        # B x T x C -> T x B x C

        x = x.transpose(0, 1)
        if suffix_x != None:
            suffix_x = suffix_x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        print("preparing for second forward pass")
        for idx, layer in enumerate(self.layers):
            # print(f"layer: {idx}  in forward pass")
            if self.prompt_insert_mode == 1:
                suffix_x = None
            if self.adapter_arch == 'compacter':
                if idx + 1 >= self.adapter_insert_layer:
                    compacter_n=self.compacter_n
                    tmp_id = idx + 1 - self.adapter_insert_layer
                    tmp_idx = (idx + 1 - self.adapter_insert_layer) * compacter_n * 2
                    
                    x, _ = layer(x,
                                 self_attn_padding_mask=padding_mask,
                                 compacter_n=self.compacter_n,
                                 suffix_x=suffix_x,
                                 adapter_arch=self.adapter_arch,
                                 compacter_down_proj_s=self.compacter_down_proj_s[tmp_idx:tmp_idx+compacter_n],
                                 compacter_down_proj_t=self.compacter_down_proj_t[tmp_idx:tmp_idx+compacter_n],
                                 compacter_down_proj_b=self.compacter_down_proj_b[tmp_id*2],
                                 compacter_up_proj_s=self.compacter_up_proj_s[tmp_idx:tmp_idx+compacter_n],
                                 compacter_up_proj_t=self.compacter_up_proj_t[tmp_idx:tmp_idx+compacter_n],
                                 compacter_up_proj_b=self.compacter_up_proj_b(torch.arange(tmp_id*2,tmp_id*2+1).type_as(tokens)),
                                 compacter_shared_A=self.compacter_shared_A[tmp_idx:tmp_idx+compacter_n],
                                 compacter_down_proj_s2=self.compacter_down_proj_s[tmp_idx+compacter_n:tmp_idx+compacter_n*2],
                                 compacter_down_proj_t2=self.compacter_down_proj_t[tmp_idx+compacter_n:tmp_idx+compacter_n*2],
                                 compacter_down_proj_b2=self.compacter_down_proj_b[tmp_id*2+1],
                                 compacter_up_proj_s2=self.compacter_up_proj_s[tmp_idx+compacter_n:tmp_idx+compacter_n*2],
                                 compacter_up_proj_t2=self.compacter_up_proj_t[tmp_idx+compacter_n:tmp_idx+compacter_n*2],
                                 compacter_up_proj_b2=self.compacter_up_proj_b(torch.arange(tmp_id*2+1,tmp_id*2+2).type_as(tokens)),
                                 compacter_shared_A2=self.compacter_shared_A[tmp_idx+compacter_n:tmp_idx+compacter_n*2],)
            else:

                # if we are at last layer before the adapter insert create adapter insert otherwise done
                if idx + 1 >= self.adapter_insert_layer:
                    if self.adapter_arch == 'houlsby':
                        adapter_MLP2 = self.middle_adapter_MLP2[idx + 1 - self.adapter_insert_layer]
                    else:
                        adapter_MLP2 = None
                    adapter_MLP = self.middle_adapter_MLP[idx + 1 - self.adapter_insert_layer]
                else:
                    adapter_MLP = None
                    adapter_MLP2 = None

                torch.cuda.empty_cache()
                # look into how the suffix x affects the forward pass
                # if not x.is_cuda:
                #     x = x.to(f"cuda:{self.args.device_id}")
                x, _ = layer(x, self_attn_padding_mask=padding_mask, suffix_x=suffix_x, adapter_MLP=adapter_MLP, adapter_MLP2=adapter_MLP2, adapter_arch=self.adapter_arch)
                # x = x.to("cpu")
                torch.cuda.empty_cache()

            # if we are not at the adapter insert layer
            if idx + 1 >= self.middle_prompt_insert_layer and idx < 23:
                # middle_prompt_insert_layer is between [1, 23]
                # layer 0 is already done
                # layer 24 has no affect with cls token, so no need to insert
                if self.sentence_generation is not None: #we enter here
                    if self.middle_previous:  # true so we enter here. 
                        
                        # first token for all batches and rep dimensions
                        # shape B X C
                        suffix_x = x[0, :, :].clone()
                    else:
                        suffix_x = suffix_x0.clone()

                    torch.cuda.empty_cache()
                    if self.generation_quaternions is not None:
                        if self.middle_prompt_mode == 'none':
                            tmp_x = self.middle_generation_dense_b[idx].repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                            # 16 x 256
                            for i in range(self.generation_quaternions):
                                if self.lphm is None:
                                    tmp_x += torch.mm(suffix_x, kron(self.middle_generation_shared_w1[idx * self.generation_quaternions + i], self.middle_generation_dense_w[idx * self.generation_quaternions + i]).to(f"cuda:{self.args.device_id}"))
                                else: 
                                    tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w[i], torch.mm(self.generation_dense_s[i], self.generation_dense_t[i])).to(f"cuda:{self.args.device_id}"))
                        else:
                            # for shared mode and layerb mode

                            # repeat for the batch
                            # 
                            # # shape  B x m
                            tmp_x = self.generation_dense_b.repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                            # tmp_x = self.generation_dense_b.repeat(tokens.shape[0], 1).to('cpu')
                            
                            for i in range(self.generation_quaternions):
                                if self.lphm is None:
                                    tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w1[i], self.generation_dense_w[i]).to(f"cuda:{self.args.device_id}"))
                                    # tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w1[i], self.generation_dense_w[i]).to(f"cuda:{self.args.device_id}")).to('cpu')
                                else: 
                                    tmp_x += torch.mm(suffix_x, kron(self.generation_shared_w[i], torch.mm(self.generation_dense_s[i], self.generation_dense_t[i])).to(f"cuda:{self.args.device_id}"))

                        # both shape B x M
                        torch.cuda.empty_cache()
                        # if not tmp_x.is_cuda:
                        #     tmp_x = tmp_x.to(f"cuda:{self.args.device_id}")
                        suffix_x1 = self.generation_activation_fn(tmp_x)
                        suffix_x2 = self.generation_dropout(suffix_x1)
                        # 16 x 256


                        if self.middle_prompt_mode == 'none' or self.middle_prompt_mode == 'layerb':

                            # shape B x (suffix_len * d)
                            # tmp_x1 = self.middle_generation_out_proj_b(torch.arange(idx * self.suffix_len * self.prompt_insert_mode, (idx + 1) * self.suffix_len * self.prompt_insert_mode,).type_as(tokens)).view(-1).repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                            tmp_x1 = self.middle_generation_out_proj_b(torch.arange(idx * self.suffix_len * self.prompt_insert_mode, (idx + 1) * self.suffix_len * self.prompt_insert_mode, dtype = torch.long).to(f"cuda:{self.args.device_id}")).view(-1).repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                            # tmp_x1 = self.middle_generation_out_proj_b(torch.arange(idx * self.suffix_len * self.prompt_insert_mode, (idx + 1) * self.suffix_len * self.prompt_insert_mode, dtype = torch.long).to(f"cuda:{self.args.device_id}")).view(-1).repeat(tokens.shape[0], 1).to('cpu')
                        else:
                            # for shared mode
                            tmp_x1 = self.generation_out_proj_b(torch.arange(self.suffix_len * self.prompt_insert_mode).type_as(tokens)).view(-1).repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                        
                        if self.middle_prompt_mode == 'none':
                            for i in range(self.generation_quaternions):
                                if self.lphm is None:
                                    tmp_x1 += torch.mm(suffix_x2, kron(self.middle_generation_shared_w2[idx * self.generation_quaternions + i], self.middle_generation_out_proj_w[idx * self.generation_quaternions + i]).to(f"cuda:{self.args.device_id}"))
                                else:
                                    tmp_x1 += torch.mm(suffix_x2, kron(self.generation_shared_w[i], torch.mm(self.generation_out_proj_s[i], self.generation_out_proj_t[i])).to(f"cuda:{self.args.device_id}"))
                        else:
                            for i in range(self.generation_quaternions):
                                if self.lphm is None:
                                    tmp_x1 += torch.mm(suffix_x2, kron(self.generation_shared_w2[i], self.generation_out_proj_w[i]).to(f"cuda:{self.args.device_id}"))
                                    # tmp_x1 += torch.mm(suffix_x2, kron(self.generation_shared_w2[i], self.generation_out_proj_w[i]).to(f"cuda:{self.args.device_id}")).to("cpu")
                                else:
                                    tmp_x1 += torch.mm(suffix_x2, kron(self.generation_shared_w[i], torch.mm(self.generation_out_proj_s[i], self.generation_out_proj_t[i])).to(f"cuda:{self.args.device_id}"))
                        

                        # shape suffix_len X B x d
                        layer_prompt = tmp_x1.view(-1, self.suffix_len * self.prompt_insert_mode, self.embedding_dim).transpose(0, 1)
                        # print(layer_prompt.device, "layer prompt device")
                        torch.cuda.empty_cache()
                        # print("we have generated the layer prompt")

                        # if self.insert_position == 6:
                            # maybe we truncate

                        # 5 x 16 x 1024


                        if self.prompt_insert_mode != 2: # we wnter here 

                            if self.insert_position != 6:
                                for i in range(layer_prompt.size()[1]): # for each batch, add in the layer prompt to the input
                                    x[self.stop_idxs[i]:self.stop_idxs[i]+self.suffix_len, i] += layer_prompt[:,i]   # we add the layer prompt ontop of the input
                            else:

                                for i in range(layer_prompt.size()[1]):
                                    stop_idx = self.stop_idxs[i]
                                    for j, stop_indexes in enumerate(stop_idx):
                                        x[stop_indexes:stop_indexes+ actual_suffix_length, i] += layer_prompt[j * actual_suffix_length:(j+1)*actual_suffix_length,i]
                            torch.cuda.empty_cache()
                            # print('we have added the layer prompt to x')
                        else:
                            suffix_x = layer_prompt
                    else:


                        if self.middle_prompt_mode == 'layerb':
                            suffix_x = self.generation_dense(suffix_x)
                            suffix_x = self.generation_activation_fn(suffix_x)
                            suffix_x = self.generation_dropout(suffix_x)
                            if self.generation_layer == 2:
                                suffix_x = self.generation_out_proj(suffix_x)
                                if self.middle_prompt_insert_layer != 25:
                                    suffix_x += self.middle_generation_out_proj_b(torch.arange((idx + 1) * self.suffix_len * self.prompt_insert_mode, (idx + 2) * self.suffix_len * self.prompt_insert_mode).type_as(tokens)).view(-1).repeat(tokens.shape[0], 1).to(f"cuda:{self.args.device_id}")
                        else:
                            #suffix_x = torch.ones(suffix_x.size()[0], suffix_x.size()[1]).to(f"cuda:{self.args.device_id}")
                            suffix_x = self.middle_generation_dense[22 - idx](suffix_x)
                            suffix_x = self.generation_activation_fn(suffix_x)
                            suffix_x = self.generation_dropout(suffix_x)
                            suffix_x = self.middle_generation_out_proj[22 - idx](suffix_x)


                        layer_prompt = suffix_x.view(-1, self.suffix_len * self.prompt_insert_mode, self.embedding_dim).transpose(0, 1)
                        if self.prompt_insert_mode != 2:
                            for i in range(layer_prompt.size()[1]):
                                x[self.stop_idxs[i]:self.stop_idxs[i]+self.suffix_len, i] += layer_prompt[:,i]
                        else:
                            suffix_x = layer_prompt
                else:
                    if self.middle_prompt_mode == 'none':
                        layer_prompt = self.middle_prompt(torch.arange(idx * self.suffix_len * self.prompt_insert_mode, (idx + 1) * self.suffix_len * self.prompt_insert_mode).type_as(tokens)).repeat(tokens.shape[0], 1).view(-1, self.suffix_len * self.prompt_insert_mode, self.embedding_dim).transpose(0, 1)

                        if self.prompt_insert_mode != 2:
                            for i in range(layer_prompt.size()[1]):
                                x[self.stop_idxs[i]:self.stop_idxs[i]+self.suffix_len, i] += layer_prompt[:,i]
                        else:
                            suffix_x = layer_prompt
                    elif self.middle_prompt_mode == 'adapter-shared':
                        print('yes')
                    else:
                        print('yes')

            if not last_state_only:
                inner_states.append(x)

        print("we got past second forward pass")

        torch.cuda.empty_cache()
        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
