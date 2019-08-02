__author__ = 'Wenlu Wang'

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_manager import load_data
from embed import _embed_list
import numpy as np
import tensorflow as tf
import sys
import glove
import argparse
# ----------------------------------------------------------------------------

maxlen0 = 20
maxlen1 = 2
embedding_dim = 300
batch_size = 2
train_batch_size = 2
n_states = 300
classes = 2
train_needed = True
train_epochs = 20
# ----------------------------------------------------------------------------

def matcher(text, key):
    """Match LSTM.

    The match LSTM could be thought as encoding text according to the
    attention weight, w.r.t key.

    :param text: [B, T0, D]
    :param key: [B, T1, D]

    :returns: [B, T1, D]
    """
    key_shape = tf.shape(key)
    print('key shape:')
    print(key.get_shape().as_list())
    B, T1 = key_shape[0], key_shape[1]

    const_key_shape = key.get_shape().as_list()
    D = const_key_shape[-1]

    cell = tf.nn.rnn_cell.LSTMCell(D)

    # [B, T1, D] => [T1, B, D]
    key = tf.transpose(key, [1, 0, 2])

    w0 = tf.get_variable('w0', [D, D], tf.float32)
    w1 = tf.get_variable('w1', [D, D], tf.float32)
    w2 = tf.get_variable('w2', [D, D], tf.float32)
    v0 = tf.get_variable('v0', [1, D], tf.float32)
    v1 = tf.get_variable('v1', [D, 1], tf.float32)
    s0 = tf.get_variable('s0', (), tf.float32)

    a = tf.tensordot(text, w0, [[2], [0]])
    b = tf.tensordot(key, w1, [[2], [0]])

    def _cond(i, *_):
        return tf.less(i, T1)

    def _body(i, outputs, state):
        b_i = tf.expand_dims(tf.gather(b, i), axis=1)
        key_i = tf.gather(key, i)

        # [B, 1, D]
        c = tf.expand_dims(tf.matmul(state[1], w2) + v0, axis=1)
        # [B, T0, 1]
        d = tf.tensordot(tf.tanh(a + b_i + c), v1, [[2], [0]]) + s0
        # [B, 1, T0]
        attn = tf.nn.softmax(tf.transpose(d, [0, 2, 1]))
        # [B, D]
        ciphertext = tf.squeeze(tf.matmul(attn, text), axis=1)
        # [B, 2D]
        e = tf.concat((key_i, ciphertext), axis=1)

        output, new_state = cell(e, state)
        outputs = outputs.write(i, output)

        return i + 1, outputs, new_state

    outputs = tf.TensorArray(tf.float32, T1)
    state = cell.zero_state(B, tf.float32)
    _, outputs, _ = tf.while_loop(_cond, _body, (0, outputs, state),
                                  name='matcher')

    # [T1, B, D] => [B, T1, D]
    outputs = tf.transpose(outputs.stack(), [1, 0, 2])
    outputs.set_shape(const_key_shape)

    return outputs



# ------------------------------------------------------------------------------

def train(sess, env, X0_data, X1_data, y_data, X0_valid, X1_valid, y_valid ,epochs=1, load=False,
          shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X0_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        #print('\nEpoch {0}/{1}'.format(epoch+1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X0_data = X0_data[ind]
            X1_data = X1_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            #print(' batch {0}/{1}'.format(batch+1, n_batch))
            sys.stdout.flush()
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            cnt = end - start
            if cnt < batch_size:
                break
            sess.run(env.train_op, feed_dict={env.x0: X0_data[start:end],
                                              env.x1: X1_data[start:end],
                                              env.y: y_data[start:end]})
        evaluate(sess, env, X0_valid, X1_valid, y_valid, batch_size=batch_size)

    print('\n Saving model')
    #os.makedirs('model', exist_ok=True)
    env.saver.save(sess, 'model/{}'.format(name))

def evaluate(sess, env, X0_data, X1_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X0_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0
    for batch in range(n_batch):
        #print(' batch {0}/{1}'.format(batch+1, n_batch))
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        if cnt < batch_size:
            break
        batch_loss, batch_acc, yy = sess.run(
            [env.loss, env.acc, env.ybar],
            feed_dict={env.x0: X0_data[start:end],
                       env.x1: X1_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
        #print(yy)
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

def inference(sess, env, X0_data, X1_data):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    tru_len = len(X0_data)
    X0_data = np.vstack([X0_data, np.zeros((30-X0_data.shape[0], maxlen0, embedding_dim))])
    X1_data = np.vstack([X1_data, np.zeros((30-X1_data.shape[0], maxlen1, embedding_dim))])
    ybar = sess.run(env.ybar, feed_dict={env.x0: X0_data, env.x1: X1_data})
    ybar = ybar[:tru_len]
            
    return np.argmax(ybar), ybar

# ------------------------------------------------------------------------------
class Dummy:
    pass


def build_model(env):
    # Convert to internal representation

    cell0 = tf.nn.rnn_cell.LSTMCell(n_states)
    H0, _ = tf.nn.dynamic_rnn(cell0, env.x0, dtype=tf.float32, scope='h0')

    cell1 = tf.nn.rnn_cell.LSTMCell(n_states)
    H1, _ = tf.nn.dynamic_rnn(cell1, env.x1, dtype=tf.float32, scope='h1')
    print('H1 shape')
    print(H1.get_shape().as_list())
    with tf.variable_scope('fw') as scope:
        outputs_fw = matcher(H0, H1)

    with tf.variable_scope('bw') as scope:
        outputs_bw = matcher(H0, tf.reverse(H1, axis=[1]))

    outputs = tf.concat((outputs_fw, outputs_bw), axis=-1, name='h0h1')
    print(outputs.get_shape().as_list())
    output = tf.reduce_mean(outputs, axis=1)

    layer1 = tf.layers.dense(output, 200)
    #dr1 = tf.layers.dropout(layer1, rate=.5)
    layer2 = tf.layers.dense(layer1, 100)
    #dr2 = tf.layers.dropout(layer2, rate=.5)

    logits = tf.layers.dense(layer2, 1)
    env.ybar = tf.sigmoid(logits)

    with tf.variable_scope('loss'):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=env.y,logits=logits)
        env.loss = tf.reduce_mean(xent)

    with tf.variable_scope('acc'):
        t0 = tf.greater(env.ybar, 0.5)
        t1 = tf.greater(env.y, 0.5)
        count = tf.equal(t0, t1)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    optimizer = tf.train.AdamOptimizer()
    env.train_op = optimizer.minimize(env.loss)

    #env.dy_dx, = tf.gradients(env.loss, env.x0)

    env.saver = tf.train.Saver()

# ------------------------------------------------------------------------------

# for v in tf.trainable_variables():
#     print(v.name)

# for v in tf.get_default_graph().as_graph_def().node:
#     print(v.name)

# ------------------------------------------------------------------------------
class TF:
    sess = None
    env = None
    def __init__(self):
        batch_size = 30
        self.env = Dummy()
        self.env.x0 = tf.placeholder(tf.float32, (batch_size, maxlen0, embedding_dim),
                                name='x0')
        self.env.x1 = tf.placeholder(tf.float32, (batch_size, maxlen1, embedding_dim),
                            name='x1')
        self.env.y = tf.placeholder(tf.float32, (batch_size, 1), name='y')
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        build_model(self.env)
        self.env.saver.restore(self.sess, "/nfs_shares/jzl0166_home/binary_classifier/model/word_model")

    def infer(self, ls, g=None):
        if ls == []:
            ls = ['how many rivers are found in <f0> colorado <eof>\tcity', 'how many rivers are found in <f0> colorado <eof>\tstate', 'how many rivers are found in <f0> colorado <eof>\triver']       
 
        X_inf_qu, X_inf_col = _embed_list(ls, g)

        res, ybar = inference(self.sess, self.env, X_inf_qu, X_inf_col)
        idxs = np.argwhere(ybar>.5)
        match = ''
        if len(idxs)>1:
            for i, idx in enumerate(idxs): 
                idx = idx[0]
                token = ls[idx].split('\t')[1]
                if '<eof> ' + token in ls[idx]:
                    match = token
        if not match:        
            match = ls[res].split('\t')[1]
        
        return match, ybar

if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--mode',
        choices=['train', 'infer'],
        default='infer',
        help='Run mode')
    args = arg_parser.parse_args()
    
    if False:
        ''' infer a question '''
        tf_model = TF()
        g = glove.Glove()
        flag, prob = tf_model.infer([], g)
        
    else:
        ''' train/infer '''
        env = Dummy()

        env.x0 = tf.placeholder(tf.float32, (batch_size, maxlen0, embedding_dim),
                                name='x0')
        env.x1 = tf.placeholder(tf.float32, (batch_size, maxlen1, embedding_dim),
                                name='x1')
        env.y = tf.placeholder(tf.float32, (batch_size, 1), name='y')
        build_model(env)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Load data start...')
        X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev = load_data()
        print('Load data done...')

        if args.mode == 'train':
            train(sess, env, X_train_qu, X_train_col, y_train, X_dev_qu, X_dev_col, y_dev, epochs=train_epochs, load=False,
                          shuffle=True, batch_size=batch_size, name='word_model')
            evaluate(sess, env, X_test_qu, X_test_col, y_test, batch_size=batch_size)
        else:
            env.saver.restore(sess, "model/word_model")

        print('Train')
        evaluate(sess, env, X_train_qu, X_train_col, y_train, batch_size=batch_size)
        print('Test')
        evaluate(sess, env, X_test_qu, X_test_col, y_test, batch_size=batch_size)
        
