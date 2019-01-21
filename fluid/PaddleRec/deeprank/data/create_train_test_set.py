import pickle
import sys
import random
import os


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_to_file(data_set, path, movie_count, cat_count):
    with open(path, 'w') as wf:
        for line in data_set:
            i = 0
            for e in line:
                if len(e) == 0:
                    if i % 2 == 0:
                        wf.write(str(movie_count))
                    else:
                        wf.write(str(cat_count))
                else:
                    for w in e:
                        wf.write(str(w) + " ")
                if i < len(line) - 1:
                    wf.write(",")
                else:
                    wf.write("\n")
                i += 1


def split_solve(data_num):
    with open("remap" + data_num + ".pkl", 'rb') as f:
        ratings = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, movie_count, cate_count, example_count = pickle.load(f)
    print "user count: %d, movie count: %d, cate count: %d, example count:%d" % (
        user_count, movie_count, cate_count, example_count)
    file_path = "data_" + data_num + "m"
    train_path = file_path + "/train_data"
    test_path = file_path + "/test_data"
    create_dir_if_not_exist(file_path)
    create_dir_if_not_exist(train_path)
    create_dir_if_not_exist(test_path)
    with open(file_path + "/config.txt", "w") as wf:
        wf.write(str(user_count + 1) + "\n")
        wf.write(str(movie_count + 1) + "\n")
        wf.write(str(cate_count + 1) + "\n")

    train_set = []
    test_set = []
    num = 0
    train_block = 0
    test_block = 0
    is_train = True
    train_test_gap = user_count * 0.7
    if data_num == "20":
        train_test_gap = 100000
    for userid, ele in ratings.groupby('userId'):
        hist = ele['movieId'].tolist()
        rate = ele['rating'].tolist()

        good_hist = []
        bad_hist = []
        good_hist_cat = []
        bad_hist_cat = []

        for i in range(1, len(hist)):
            if rate[i - 1] == 1:
                good_hist.append(hist[i - 1])
                good_hist_cat.append(cate_list[hist[i - 1]])
            else:
                bad_hist.append(hist[i - 1])
                bad_hist_cat.append(cate_list[hist[i - 1]])
            if num < train_test_gap:
                train_set.append([
                    good_hist[:], good_hist_cat[:], bad_hist[:],
                    bad_hist_cat[:], [hist[i]], [cate_list[hist[i]]],
                    [rate[i]]
                ])
            else:
                test_set.append([
                    good_hist[:], good_hist_cat[:], bad_hist[:],
                    bad_hist_cat[:], [hist[i]], [cate_list[hist[i]]],
                    [rate[i]]
                ])
                #train_set.append([[userid], good_hist[:], good_hist_cat[:], bad_hist[:], bad_hist_cat[:], [hist[i]], [rate[i]]])
        num += 1
        if num % 1000 == 0:
            print "do %d ing..." % (num)
            if len(train_set) > 0:
                random.shuffle(train_set)
                path = train_path + "/block_" + str(train_block)
                write_to_file(train_set, path, movie_count, cate_count)
                train_set = []
                train_block += 1
            if len(test_set) > 0:
                random.shuffle(test_set)
                path = test_path + "/block_" + str(test_block)
                write_to_file(test_set, path, movie_count, cate_count)
                test_set = []
                test_block += 1

    if len(train_set) > 0:
        random.shuffle(train_set)
        path = train_path + "/block_" + str(train_block)
        write_to_file(train_set, path, movie_count, cate_count)
    if len(test_set) > 0:
        random.shuffle(test_set)
        path = test_path + "/block_" + str(test_block)
        write_to_file(test_set, path, movie_count, cate_count)

    print "complete successful..."


def solve(data_num):
    with open("remap" + data_num + ".pkl", 'rb') as f:
        ratings = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, movie_count, cate_count, example_count = pickle.load(f)

    train_set = []
    test_set = []

    num = 0
    gap = user_count * 0.9
    for userid, ele in ratings.groupby('userId'):
        num += 1
        if num % 10000 == 0:
            print "do %d ing..." % (num)
        hist = ele['movieId'].tolist()
        rate = ele['rating'].tolist()

        good_hist = []
        bad_hist = []
        good_hist_cat = []
        bad_hist_cat = []

        for i in range(1, len(hist)):
            if rate[i - 1] == 1:
                good_hist.append(hist[i - 1])
                good_hist_cat.append(cate_list[hist[i - 1]])
            else:
                bad_hist.append(hist[i - 1])
                bad_hist_cat.append(cate_list[hist[i - 1]])
            if num < gap:
                train_set.append([[userid], good_hist[:], good_hist_cat[:],
                                  bad_hist[:], bad_hist_cat[:], [hist[i]],
                                  [rate[i]]])
            else:
                test_set.append([[userid], good_hist[:], good_hist_cat[:],
                                 bad_hist[:], bad_hist_cat[:], [hist[i]],
                                 [rate[i]]])

    random.shuffle(train_set)
    random.shuffle(test_set)

    print "user count: %d, movie count: %d, cate count: %d, example count:%d" % (
        user_count, movie_count, cate_count, example_count)
    print train_set[:4]
    print test_set[:4]

    with open('train_data_' + data_num + '.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, movie_count, cate_count), f,
                    pickle.HIGHEST_PROTOCOL)

    with open('test_data_' + data_num + '.pkl', 'wb') as f:
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, movie_count, cate_count), f,
                    pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if sys.argv[1] == "1" or sys.argv[1] == "0":
        split_solve(sys.argv[1])
    elif sys.argv[1] == "20":
        split_solve(sys.argv[1])
