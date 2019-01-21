import sys
import pickle
import os
import time
import six
import numpy as np
import math
import argparse
import paddle.fluid as fluid
import paddle
import time
import network
import reader


def train():
    bs = 32
    data_num = sys.argv[1]
    config_path = "data/data_" + data_num + "m/config.txt"
    data_path = "data/data_" + data_num + "m/train_data"
    data_reader, user_count, movie_count, cat_count = reader.prepare_data(
        config_path, data_path)

    batch_reader = paddle.batch(
        paddle.reader.shuffle(
            data_reader, buf_size=bs * 500), batch_size=bs)

    avg_loss, auc_var, accuracy = network.network(movie_count, cat_count)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.1)
    sgd_optimizer.minimize(avg_loss)

    place = fluid.CPUPlace()
    #place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    feeder = fluid.DataFeeder(
        feed_list=[
            "hist_good_seq", "hist_good_cat_seq", "hist_bad_seq",
            "hist_bad_cat_seq", "target", "target_cat", "label"
        ],
        place=place)
    if False:
        train_exe = fluid.ParallelExecutor(
            use_cuda=True, loss_name=avg_loss.name)
    else:
        train_exe = exe

    for id in range(200):
        epoch = id + 1
        step = 0
        start_time = time.time()
        for data in batch_reader():
            step += 1
            results = train_exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_loss.name, auc_var.name, accuracy.name],
                return_numpy=True)
            if step % 20 == 0:
                print "epoch: %d, step: %d, time: %.2f" % (
                    epoch, step, time.time() - start_time)
                start_time = time.time()
                print results
            if step % 5000 == 0:
                save_dir = "deep_rank_movielens_" + data_num + "m/epoch_" + str(
                    epoch) + "step_" + str(step)
                feed_var_name = [
                    "hist_good_seq", "hist_good_cat_seq", "hist_bad_seq",
                    "hist_bad_cat_seq", "target", "target_cat", "label"
                ]
                fetch_vars = [avg_loss, auc_var, accuracy]
                fluid.io.save_inference_model(save_dir, feed_var_name,
                                              fetch_vars, train_exe)


if __name__ == "__main__":
    train()
