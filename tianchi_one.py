import tensorflow as tf
import numpy as np
import random
import pandas as pd


# define function
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_kk(x, k, name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


def drop(x, keep_p, name=None):
    return tf.nn.dropout(x, keep_prob=keep_p, name=name)


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape=shape, stddev=0.05)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial, name=name)


def net(x, keep_p, input_s, num_o):
    with tf.name_scope('net'):
        W1 = weight_variable(shape=[input_s, 2000], name='W1')
        b1 = bias_variable(shape=[2000], name='b1')
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1, name='h1')
        d1 = drop(h1, keep_p, name='d1')

        W2 = weight_variable(shape=[2000, 100], name='W2')
        b2 = bias_variable(shape=[100], name='b2')
        h2 = tf.nn.relu(tf.matmul(d1, W2) + b2, name='h2')
        d2 = drop(h2, keep_p, name='d2')
        #
        # W3 = weight_variable(shape=[500, 100], name='W3')
        # b3 = bias_variable(shape=[100], name='b3')
        # h3 = tf.nn.tanh(tf.matmul(d2, W3) + b3, name='h3')
        # d3 = drop(h3, keep_p, name='d3')

        W_out = weight_variable(shape=[100, num_o], name='W_out')
        b_out = bias_variable(shape=[num_o], name='b_out')
        y_out = tf.matmul(d2, W_out) + b_out

        return y_out


def read_data(valid_size, train_p, A_test_p, B_test_p):
    train_f = pd.read_excel(train_p)
    a_test_f = pd.read_excel(A_test_p)
    b_test_f = pd.read_excel(B_test_p)

    string_cols = list()
    for col in train_f.columns[1:-1]:
        if isinstance(train_f[col].values[0], str):
            string_cols.append(col)

    list_list = list()
    for col in string_cols:
        ls = list(set(train_f[col].values))
        train_f[col] = train_f[col].apply(lambda va: ls.index(va))
        a_test_f[col] = a_test_f[col].apply(lambda va: ls.index(va))
        b_test_f[col] = b_test_f[col].apply(lambda va: ls.index(va))
        list_list.append(ls)

    col_max_value = list()
    for col in train_f.columns[1:-1]:
        col_mean_train = np.mean([i for i in train_f[col].values if pd.notnull(i)])
        col_mean_test_a = np.mean([i for i in a_test_f[col].values if pd.notnull(i)])
        col_mean_test_b = np.mean([i for i in b_test_f[col].values if pd.notnull(i)])
        if pd.isnull(col_mean_train):
            col_mean_train = 0.0
        if pd.isnull(col_mean_test_a):
            col_mean_test_a = 0.0
        if pd.isnull(col_mean_test_b):
            col_mean_test_b = 0.0
        train_f[col] = train_f[col].apply(lambda va: col_mean_train if pd.isnull(va) else va)
        a_test_f[col] = a_test_f[col].apply(lambda va: col_mean_test_a if pd.isnull(va) else va)
        b_test_f[col] = b_test_f[col].apply(lambda va: col_mean_test_b if pd.isnull(va) else va)

        # 归一化 除以每一列最大值
        col_max = np.max(train_f[col].values)
        col_max_value.append(col_max)
        train_f[col] = train_f[col].apply(lambda va: va / col_max if col_max != 0 else va)
        a_test_f[col] = a_test_f[col].apply(lambda va: va / col_max if col_max != 0 else va)
        b_test_f[col] = b_test_f[col].apply(lambda va: va / col_max if col_max != 0 else va)

    y_cols = train_f.columns[-1]
    x_cols = train_f.columns[1:-1]

    # train的x和y
    X = np.vstack(train_f[x_cols].values)
    Y = np.vstack(train_f[y_cols].values / 3.0)

    print('total X:', np.shape(X))

    # valid
    v_X = X[:valid_size]
    v_y = Y[:valid_size]
    # train
    t_X = X[valid_size:]
    t_y = Y[valid_size:]

    print('train shape:', np.shape(t_X))
    print(t_X[0])
    print(t_y[1:5])

    # test
    a_test = np.vstack(a_test_f[x_cols].values)
    b_test = np.vstack(b_test_f[x_cols].values)
    print('a test shape:', np.shape(a_test))
    print('b test shape:', np.shape(b_test))
    print(a_test[0])
    print(b_test[0])

    #ID
    a_id = a_test_f['ID'].values
    b_id = b_test_f['ID'].values

    return v_X, v_y, t_X, t_y, a_test, b_test, a_id, b_id, list_list, col_max_value


def generate_batch_train(batch_size, t_x, t_y):
    train_index = list(range(t_x.shape[0]))
    random.shuffle(train_index)
    x, y = t_x[train_index], t_y[train_index]
    off = random.randint(0, t_x.shape[0] - batch_size)
    batch_X = x[off:(off + batch_size)]
    batch_y = y[off:(off + batch_size)]
    return batch_X, batch_y


def main():
    input_size = 8027
    num_output = 1
    valid_size = 50
    batch_size = 50
    num_steps = 5000
    # load data
    valid_X, valid_y, train_X, train_y, test_A, test_B, test_A_ID, test_B_ID, util_list, cols_max = \
        read_data(valid_size, train_path, A_test_path, B_test_path)
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, input_size], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, num_output], name='input_y')
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)

    y_pred = net(x, keep_prob, input_size, num_output)

    with tf.name_scope('loss'):
        loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_pred)))
        mse = tf.reduce_mean(tf.square(y_ - y_pred))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.exp(-loss))
        tf.summary.scalar('accuracy', accuracy)
    # saver
    saver = tf.train.Saver()
    least_loss = 10000
    best_sess = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board/tianchi', sess.graph)
        for step in range(num_steps):
            batch_X, batch_y = generate_batch_train(batch_size, train_X, train_y)
            sess.run(optimizer, feed_dict={x: batch_X, y_: batch_y, keep_prob: 0.5})
            if step % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_X, y_: batch_y, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (step, train_accuracy))
                merge_result = sess.run(merged, feed_dict={x: batch_X, y_: batch_y, keep_prob: 1.0})
                writer.add_summary(merge_result, step)
            if step % 200 == 0:
                valid_loss, valid_mse = sess.run([loss, mse], feed_dict={x: valid_X, y_: valid_y, keep_prob: 1.0})
                print('step %d, valid loss %g' % (step, valid_loss))
                print('step %d, valid mse %g' % (step, valid_mse))
                # save sess
                if valid_loss < least_loss:
                    least_loss = valid_loss
                    best_sess = sess
        print('train done')

        # predict
        a_result = best_sess.run(y_pred, feed_dict={x: test_A, keep_prob: 1.0})
        a_output_file = open(A_result_test_path, 'w')
        for i in range(len(a_result)):
            a_output_file.write('{0},{1}\n'.format(test_A_ID[i], a_result[i][0] * 3.0))
        a_output_file.close()

        b_result = best_sess.run(y_pred, feed_dict={x: test_B, keep_prob: 1.0})
        b_output_file = open(B_result_test_path, 'w')
        for i in range(len(b_result)):
            b_output_file.write('{0},{1}\n'.format(test_B_ID[i], b_result[i][0] * 3.0))
        b_output_file.close()

        print('predict done!')
        # save sess
        saver.save(best_sess, '/root/tianchi_one.ckpt')
        print('save done!')

if __name__ == '__main__':
    train_path = 'F:/python_code/tianchi_tech/TianChi_Tech/data/train.xlsx'
    A_test_path = 'F:/python_code/tianchi_tech/TianChi_Tech/data/testA.xlsx'
    B_test_path = 'F:/python_code/tianchi_tech/TianChi_Tech/data/testB.xlsx'
    A_result_test_path = 'F:/python_code/tianchi_tech/TianChi_Tech/data/testA-answertemplate.csv'
    B_result_test_path = 'F:/python_code/tianchi_tech/TianChi_Tech/data/testB-answertemplate.csv'
    main()
    # valid_X, valid_y, train_X, train_y, test_A, test_B, test_A_ID, test_B_ID, util_list, cols_max = \
    #     read_data(50, train_path, A_test_path, B_test_path)
    # batch_X, batch_y = generate_batch_train(50, train_X, train_y)
    # print(np.shape(batch_X))
    # print(np.shape(batch_y))





