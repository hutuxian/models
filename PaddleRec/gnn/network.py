#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import math
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def attention(state, adj, adj_mask, bs, h, max_uniq_len, fcname):
    """
    state: [bs, uni, h]
    adj: [bs, uni, uni]
    """
    stdv = 1.0 / h
    part1 = layers.reshape(layers.expand(state, [1, 1, max_uniq_len]), [-1, max_uniq_len * max_uniq_len, h])
    part2 = layers.expand(state, [1, max_uniq_len, 1])
    inp = layers.reshape(layers.concat([part1, part2], axis=2), [-1, max_uniq_len, max_uniq_len, 2 * h]) #[bs, max_uniq_len, max_uniq_len, 2h]
    fc = layers.fc(input=inp, name=fcname, size=1, act='leaky_relu', num_flatten_dims=3,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-stdv, high=stdv)), 
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-stdv, high=stdv)))
    fc = layers.squeeze(fc, [])
    print(part1)
    print(part2)
    print(input)
    print(fc)
    print(adj)
    weight = layers.elementwise_mul(fc, adj)
    weight = layers.elementwise_add(fc, adj_mask)
    weight = layers.softmax(weight, axis = -1)
    # layers.Print(weight, message="weight", summarize=-1)
    # weight = layers.dropout(weight, 0.6, is_test=True, dropout_implementation='upscale_in_train')
    ret_state = layers.matmul(weight, state) 

    print(weight)
    print(ret_state)
    return ret_state


def network(items_num, hidden_size, step, bs, max_uniq_len=70, attdim=8):
    stdv = 1.0 / math.sqrt(hidden_size)

    items = fluid.data(
        name="items",
        shape=[bs, max_uniq_len],
        dtype="int64") #[batch_size, uniq_max]
    pos = fluid.data(
        name="pos",
        shape=[bs, -1],
        dtype="int64") #[batch_size, uniq_max, 1]
    seq_index = fluid.data(
        name="seq_index",
        shape=[bs, -1, 2],
        dtype="int32") #[batch_size, seq_max, 2]
    last_index = fluid.data(
        name="last_index",
        shape=[bs, 2],
        dtype="int32") #[batch_size, 2]
    adj_in = fluid.data(
        name="adj_in",
        shape=[bs, max_uniq_len, max_uniq_len],
        dtype="float32") #[batch_size, seq_max, seq_max]
    adj_out = fluid.data(
        name="adj_out",
        shape=[bs, max_uniq_len, max_uniq_len],
        dtype="float32") #[batch_size, seq_max, seq_max]
    adj_in_mask = fluid.data(
        name="adj_in_mask",
        shape=[bs, max_uniq_len, max_uniq_len],
        dtype="float32") #[batch_size, seq_max, seq_max]
    adj_out_mask = fluid.data(
        name="adj_out_mask",
        shape=[bs, max_uniq_len, max_uniq_len],
        dtype="float32") #[batch_size, seq_max, seq_max]
    mask = fluid.data(
        name="mask",
        shape=[bs, -1, 1],
        dtype="float32") #[batch_size, seq_max, 1]
    label = fluid.data(
        name="label",
        shape=[bs, 1],
        dtype="int64") #[batch_size, 1]

    datas = [items, pos, seq_index, last_index, adj_in, adj_out, adj_in_mask, adj_out_mask, mask, label]
    py_reader = fluid.io.DataLoader.from_generator(capacity=256, feed_list=datas, iterable=False)
    feed_datas = datas

    items_emb = fluid.embedding(
        input=items,
        padding_idx=0,
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[batch_size, uniq_max, h]
    pos_emb = fluid.embedding(
        input=pos,
        param_attr=fluid.ParamAttr(
            name="pos_emb",
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[1000, hidden_size])  #[batch_size, uniq_max, h]

    pre_state = items_emb
    state_list = []
    atten_h = 8
    for i in range(step):
        for j in range(8): # nheads
            state_input = layers.reshape(x=pre_state, shape=[bs, -1, hidden_size])
            state_in = layers.fc(
                input=state_input,
                name="state_in"+str(j),
                size=atten_h,
                act=None,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
            state_out = layers.fc(
                input=state_input,
                name="state_out"+str(j),
                size=atten_h,
                act=None,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
            
            att_in = attention(state_in, adj_in, adj_in_mask, bs, atten_h, max_uniq_len, "fc_att_in"+str(j)) #[batch_size, uniq_max, h]
            att_out = attention(state_out, adj_out, adj_out_mask, bs, atten_h, max_uniq_len, "fc_att_out"+str(j))
            concat_att = layers.concat([att_in, att_out], axis=2)
            print("concat_att: ", concat_att)
            # state_output = layers.fc(concat_att, name="att_out", size=hidden_size, num_flatten_dims=2)
            state_list.append(concat_att)

            # state_adj_in = layers.matmul(adj_in, state_in)  #[batch_size, uniq_max, h]
            # state_adj_out = layers.matmul(adj_out, state_out)   #[batch_size, uniq_max, h]
            # gru_input = layers.concat([state_adj_in, state_adj_out], axis=2)
        state_list_concat = layers.concat(state_list, axis=2)
        print(state_list_concat)
        gru_input = layers.fc(state_list_concat, name="att_out", size=hidden_size*2, num_flatten_dims=2)
        gru_input = layers.reshape(x=gru_input, shape=[-1, hidden_size * 2])
        gru_fc = layers.fc(
            input=gru_input,
            name="gru_fc",
            size=3 * hidden_size,
            bias_attr=False)
        pre_state, _, _ = fluid.layers.gru_unit(
            input=gru_fc,
            hidden=layers.reshape(x=pre_state, shape=[-1, hidden_size]),
            size=3 * hidden_size)
    # state_list_concat = layers.concat(state_list, axis=2)
    # print(state_list_concat)
    # pre_state = layers.fc(input=state_list_concat, name="att_outlist", size=hidden_size, act='elu', num_flatten_dims=2,
    #     param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-stdv, high=stdv)), 
    #     bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-stdv, high=stdv)))
    # pre_state = layers.dropout(pre_state, 0.6, is_test=True, dropout_implementation='upscale_in_train')

    final_state = layers.reshape(pre_state, shape=[bs, -1, hidden_size])
    seq = layers.gather_nd(final_state, seq_index)
    last = layers.gather_nd(final_state, last_index)

    seq += pos_emb
    seq_fc = layers.fc(
        input=seq,
        name="seq_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, seq_max, h]
    last_fc = layers.fc(
        input=last,
        name="last_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        num_flatten_dims=1,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[bathc_size, h]

    seq_fc_t = layers.transpose(
        seq_fc, perm=[1, 0, 2])  #[seq_max, batch_size, h]
    add = layers.elementwise_add(
        seq_fc_t, last_fc)  #[seq_max, batch_size, h]
    b = layers.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        default_initializer=fluid.initializer.Constant(value=0.0))  #[h]
    add = layers.elementwise_add(add, b)  #[seq_max, batch_size, h]

    add_sigmoid = layers.sigmoid(add) #[seq_max, batch_size, h] 
    add_sigmoid = layers.transpose(
        add_sigmoid, perm=[1, 0, 2])  #[batch_size, seq_max, h]

    weight = layers.fc(
        input=add_sigmoid,
        name="weight_fc",
        size=1,
        act=None,
        num_flatten_dims=2,
        bias_attr=False,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, seq_max, 1]
    weight *= mask
    weight_mask = layers.elementwise_mul(seq, weight, axis=0) #[batch_size, seq_max, h]
    global_attention = layers.reduce_sum(weight_mask, dim=1) #[batch_size, h]

    final_attention = layers.concat(
        [global_attention, last], axis=1)  #[batch_size, 2*h]
    final_attention_fc = layers.fc(
        input=final_attention,
        name="final_attention_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, h]

    all_vocab = layers.create_global_var(
        shape=[items_num - 1],
        value=0,
        dtype="int64",
        persistable=True,
        name="all_vocab")

    all_emb = fluid.embedding(
        input=all_vocab,
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[all_vocab, h]

    logits = layers.matmul(
        x=final_attention_fc, y=all_emb,
        transpose_y=True)  #[batch_size, all_vocab]
    softmax = layers.softmax_with_cross_entropy(
        logits=logits, label=label)  #[batch_size, 1]
    loss = layers.reduce_mean(softmax)  # [1]
    acc = layers.accuracy(input=logits, label=label, k=20)
    return loss, acc, py_reader, feed_datas
