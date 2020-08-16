#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import random
import copy

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    default='sample',
    help='dataset name: diginetica/yoochoose/sample')
parser.add_argument(
    '--aug_cmd',
    default='',
    help='')
parser.add_argument(
    '--aug_prob',
    default='',
    help='')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(
                    time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)  # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(
    tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
tes_sess = sorted(
    tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
print("train dataset session number: ", len(tra_sess))  # 186670    # 7966257
print("test dataset session number: ", len(tes_sess))  # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print("dict size: ", item_ctr)  # 43098, 37484
    with open("./diginetica/config.txt", "w") as fout:
        fout.write(str(item_ctr) + "\n")
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

def data_aug(data_seq, data_label, aug_cmd=[], aug_prob=[]):
    out_seqs = []
    out_labels = []
    dict_num = len(item_dict)
    assert len(data_seq) == len(data_label), "len of data_seq and data_label must be same"
    assert len(aug_cmd) == len(aug_prob), "len of cmd and prob must be same"
    supported_cmd = ["replace", "insert", "delete", "swap"]
    for aug in aug_cmd:
        assert aug in supported_cmd, "aug_cmd %s is not supported" % (aug)
    for i in range(len(aug_cmd)):
        print("%s with prob %f" % (aug_cmd[i], aug_prob[i]))
    for index in range(len(data_seq)):
        seq = data_seq[index]
        label = data_label[index]
        out_seqs += [seq]
        out_labels += [label]
        for i in range(len(aug_cmd)):
            need_aug = random.uniform(0,1) < aug_prob[i]
            if not need_aug:
                continue
            new_seq = None
            if aug_cmd[i] == "replace":
                new_seq = copy.deepcopy(seq)
                replace_id = random.randint(0, len(new_seq)-1)
                new_seq[replace_id] = random.randint(0, dict_num-1)
            elif aug_cmd[i] == "insert":
                new_seq = copy.deepcopy(seq)
                insert_id = random.randint(0, len(new_seq))
                new_seq.insert(insert_id, random.randint(0, dict_num-1))
            elif aug_cmd[i] == "delete":
                if len(seq) == 1:
                    continue
                new_seq = copy.deepcopy(seq)
                delete_id = random.randint(0, len(new_seq)-1)
                del new_seq[delete_id]
            elif aug_cmd[i] == "swap":
                new_seq = copy.deepcopy(seq)
                i = random.randint(0, len(new_seq)-1)
                j = random.randint(0, len(new_seq)-1)
                tmp = new_seq[i]
                new_seq[i] = new_seq[j]
                new_seq[j] = tmp
            out_seqs += [new_seq]
            out_labels += [label]
    return out_seqs, out_labels

aug_cmd = opt.aug_cmd.strip().split('+')
aug_prob = [float(e) for e in opt.aug_prob.strip().split('+')]
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
print("train data len before aug: %d" % (len(tr_seqs)))
tr_seqs, tr_labs = data_aug(tr_seqs, tr_labs, aug_cmd, aug_prob)
print("train data len after aug: %d" % (len(tr_seqs)))
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
# tra = (aug_train_seq, aug_train_label)
tes = (te_seqs, te_labs)

print("test data len :", len(te_seqs))
# print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
# print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    aug_info = "_".join(aug_cmd)
    prob_info = "_".join([str(e) for e in aug_prob])
    prefix = 'diginetica'
    if aug_info != "":
        prefix = 'diginetica' + "_" + aug_info + "_" + prob_info
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    pickle.dump(tra, open(prefix + '/train.txt', 'wb'))
    pickle.dump(tes, open(prefix + '/test.txt', 'wb'))
    # pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:],
                                                           tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
