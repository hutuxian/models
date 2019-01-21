import os
import paddle
import pickle


def train_reader(bs=16):
    print "begin prepare_data"
    return prepare_data()


def make_list(s):
    l = s.strip().split()
    res = [int(x) for x in l]
    return res


def data_reader(data_file_dir):
    def reader():
        filelist = os.listdir(data_file_dir)
        for file in filelist:
            with open(data_file_dir + "/" + file, "r") as fin:
                for line in fin:
                    line = line.strip().split(',')
                    good_h = make_list(line[0])
                    good_h_cat = make_list(line[1])
                    bad_h = make_list(line[2])
                    bad_h_cat = make_list(line[3])
                    target = int(line[4])
                    target_cat = int(line[5])
                    label = int(line[6])
                    yield good_h, good_h_cat, bad_h, bad_h_cat, target, target_cat, label

    return reader


def prepare_data(config_path, data_file_dir):
    with open(config_path, "r") as fin:
        user_count = int(fin.readline().strip())
        movie_count = int(fin.readline().strip())
        cat_count = int(fin.readline().strip())
    print user_count, movie_count, cat_count

    return data_reader(data_file_dir), user_count, movie_count, cat_count
