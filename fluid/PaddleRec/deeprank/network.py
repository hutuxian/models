import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle


def network(movie_count, cat_count):
    """network definition"""
    hist_emb_size = 32
    cat_emb_size = 8
    is_sparse = True
    #user = fluid.layers.data(name="user_id", shape=[1], dtype="int64") //will be added later
    hist_good_seq = fluid.layers.data(
        name="hist_good_seq", shape=[1], dtype="int64", lod_level=1)
    hist_good_cat_seq = fluid.layers.data(
        name="hist_good_cat_seq", shape=[1], dtype="int64", lod_level=1)
    hist_bad_seq = fluid.layers.data(
        name="hist_bad_seq", shape=[1], dtype="int64", lod_level=1)
    hist_bad_cat_seq = fluid.layers.data(
        name="hist_bad_cat_seq", shape=[1], dtype="int64", lod_level=1)
    target = fluid.layers.data(name="target", shape=[1], dtype="int64")
    target_cat = fluid.layers.data(name="target_cat", shape=[1], dtype="int64")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    """
    fluid.layers.Print(hist_good_seq)
    fluid.layers.Print(hist_good_cat_seq)
    fluid.layers.Print(hist_bad_seq)
    fluid.layers.Print(hist_bad_cat_seq)
    fluid.layers.Print(label)
    fluid.layers.Print(target)
    """
    """
    user_emb = fluid.layers.embedding(
        input=user,
        size=[user_count, emb_size],
        param_attr="user_emb",
        is_sparse=is_sparse)
    """

    hist_good_emb = fluid.layers.embedding(
        input=hist_good_seq,
        size=[movie_count, hist_emb_size],
        param_attr="movie_emb",
        is_sparse=is_sparse)
    hist_good_bow = fluid.layers.sequence_pool(
        input=hist_good_emb, pool_type="average")

    hist_good_cat_emb = fluid.layers.embedding(
        input=hist_good_cat_seq,
        size=[cat_count, cat_emb_size],
        param_attr="cat_emb",
        is_sparse=is_sparse)
    hist_good_cat_bow = fluid.layers.sequence_pool(
        input=hist_good_cat_emb, pool_type="average")

    hist_bad_emb = fluid.layers.embedding(
        input=hist_bad_seq,
        size=[movie_count, hist_emb_size],
        param_attr="movie_emb",
        is_sparse=is_sparse)
    hist_bad_bow = fluid.layers.sequence_pool(
        input=hist_bad_emb, pool_type="average")

    hist_bad_cat_emb = fluid.layers.embedding(
        input=hist_bad_cat_seq,
        size=[cat_count, cat_emb_size],
        param_attr="cat_emb",
        is_sparse=is_sparse)
    hist_bad_cat_bow = fluid.layers.sequence_pool(
        input=hist_bad_cat_emb, pool_type="average")

    target_emb = fluid.layers.embedding(
        input=target,
        size=[movie_count, hist_emb_size],
        param_attr="movie_emb",
        is_sparse=is_sparse)
    target_cat_emb = fluid.layers.embedding(
        input=target_cat,
        size=[cat_count, cat_emb_size],
        param_attr="cat_emb",
        is_sparse=is_sparse)

    embedding_concat = fluid.layers.concat(
        [
            hist_good_bow, hist_good_cat_bow, hist_bad_bow, hist_bad_cat_bow,
            target_emb, target_cat_emb
        ],
        axis=1)

    fc1 = fluid.layers.fc(name="fc1",
                          input=embedding_concat,
                          size=1024,
                          act="relu")
    fc2 = fluid.layers.fc(name="fc2", input=fc1, size=512, act="relu")
    fc3 = fluid.layers.fc(name="fc3", input=fc2, size=2, act="softmax")
    cost = fluid.layers.cross_entropy(input=fc3, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=fc3, label=label)
    auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=fc3,
                                                          label=label,
                                                          num_thresholds=200,
                                                          slide_steps=20)
    return avg_cost, auc_var, accuracy
