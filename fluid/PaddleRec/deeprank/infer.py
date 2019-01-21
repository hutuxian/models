import paddle.fluid as fluid
import paddle
import reader
import time
import numpy as np
import sys


def infer(test_reader, model_path):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.core.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            model_path, exe)

        step_id = 0
        feeder = fluid.DataFeeder(
            feed_list=[
                "hist_good_seq", "hist_good_cat_seq", "hist_bad_seq",
                "hist_bad_cat_seq", "target", "target_cat", "label"
            ],
            place=place,
            program=infer_program)
        for data in test_reader():
            step_id += 1
            para = exe.run(infer_program,
                           feed=feeder.feed(data),
                           fetch_list=fetch_vars,
                           return_numpy=True)

            if step_id % 20 == 0:
                print "batch:%d:auc:%lf,accurary:%lf" % (step_id, para[1],
                                                         para[2])
                #print "batch:%d:auc:%lf,accurary:%lf" % (step_id, para[1]._get_float_element(0), para[2]._get_float_element(0))


if __name__ == "__main__":
    start_index = 1
    end_index = 2
    bs = 32
    data_num = sys.argv[1]
    config_path = "data/data_" + data_num + "m/config.txt"
    data_path = "data/data_" + data_num + "m/test_data"
    data_reader, user_count, movie_count, cat_count = reader.prepare_data(
        config_path, data_path)

    batch_reader = paddle.batch(
        paddle.reader.shuffle(
            data_reader, buf_size=bs * 500), batch_size=bs)
    for epoch in range(start_index, end_index):
        print "epoch:%d" % (epoch)
        epoch_path = "deep_rank_movielens_" + data_num + "m/epoch_" + str(
            epoch) + "step_20000"
        infer(test_reader=batch_reader, model_path=epoch_path)
