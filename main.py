__author__ = 'Wenlu Wang'

# coding=utf-8
import sys
import argparse

import os
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.data_manager import load_data
from utils.bleu import moses_multi_bleu
from collections import defaultdict
from argparse import ArgumentParser

import sys
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tf_helper import train, evaluate, decode_data, decode_data_recover
from model import construct_graph


def init_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--data_path',
        default=os.path.dirname(os.path.abspath(__file__)) + '/data',
        type=str,
        help='Data path.')
    arg_parser.add_argument(
        '--load_data', default=False, type=bool, help='Load data.')
    arg_parser.add_argument(
        '--data',
        choices=['geo', 'rest'],
        default='geo',
        help='data to train & test')
    arg_parser.add_argument(
        '--subset', choices=['all'], default='all', help='Subset of data.')
    arg_parser.add_argument(
        '--maxlen', default=60, type=int, help='Data record max length.')

    arg_parser.add_argument(
        '--annotation_path',
        default=os.path.dirname(os.path.abspath(__file__)) +
        '/data/DATA/geo/',
        type=str,
        help='Data annotation path.')
    arg_parser.add_argument(
        '--mode',
        choices=['train', 'infer'],
        default='infer',
        help='Run mode')
    #### Model configuration ####
    arg_parser.add_argument(
        '--cell',
        choices=['gru'],
        default='gru',
        help='Type of cell used, currently only standard GRU cell is supported'
    )
    arg_parser.add_argument(
        '--output_vocab_size',
        default=20637,
        type=int,
        help='Output vocabulary size.')
    # Embedding sizes
    arg_parser.add_argument(
        '--embedding_dim',
        default=200,
        type=int,
        help='Size of word embeddings')

    #Hidden sizes
    arg_parser.add_argument(
        '--dim', default=200, type=int, help='Size of GRU hidden states')
    arg_parser.add_argument(
        '--hidden_size',
        default=256,
        type=int,
        help='Size of LSTM hidden states')

    arg_parser.add_argument(
        '--no_copy',
        default=False,
        action='store_true',
        help='Do not use copy mechanism')

    #### Training ####
    arg_parser.add_argument(
        '--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument(
        '--glove_embed_path',
        default=None,
        type=str,
        help='Path to pretrained Glove mebedding')

    arg_parser.add_argument(
        '--batch_size', default=128, type=int, help='Batch size')
    arg_parser.add_argument(
        '--in_drop', default=0., type=float, help='In dropout rate')
    arg_parser.add_argument(
        '--out_drop', default=0., type=float, help='Out dropout rate')

    # training details
    arg_parser.add_argument(
        '--valid_epoch_interval',
        default=1,
        type=int,
        help='Perform validation every x epoch')
    arg_parser.add_argument(
        '--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument(
        '--total_epochs', default=30, type=int, help='# of training epoches')
    arg_parser.add_argument(
        '--epochs', default=20, type=int, help='Record per x epoches')
    arg_parser.add_argument(
        '--lr', default=0.0001, type=float, help='Learning rate')
    arg_parser.add_argument(
        '--lr_decay',
        default=0.5,
        type=float,
        help='decay learning rate if the validation performance drops')

    #### decoding/validation/testing ####
    arg_parser.add_argument(
        '--load_model', default=False, type=bool, help='Whether to load model')
    arg_parser.add_argument(
        '--beam_width', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument(
        '--decode_max_time_step',
        default=100,
        type=int,
        help='Maximum number of time steps used '
        'in decoding and sampling')
    args = arg_parser.parse_args()
    return args

def model(args, train_env, infer_env):

    tf.reset_default_graph()
    train_graph = tf.Graph()
    infer_graph = tf.Graph()

    with train_graph.as_default():
        train_env.x = tf.placeholder(
            tf.int32, shape=[None, args.maxlen], name='x')
        train_env.y = tf.placeholder(tf.int32, (None, args.maxlen), name='y')
        train_env.training = tf.placeholder_with_default(
            False, (), name='train_mode')
        train_env.train_op, train_env.loss, train_env.acc, sample_ids, logits = construct_graph(
            "train", train_env, args)
        #train_env.saver = tf.train.Saver()
        train_env.saver = tf.train.Saver(max_to_keep=None)
        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node if 'xxxxx' in n.name]

    with infer_graph.as_default():
        infer_env.x = tf.placeholder(
            tf.int32, shape=[None, args.maxlen], name='x')
        infer_env.y = tf.placeholder(tf.int32, (None, args.maxlen), name='y')
        infer_env.training = tf.placeholder_with_default(
            False, (), name='train_mode')
        _, infer_env.loss, infer_env.acc, infer_env.pred_ids, _ = construct_graph(
            "infer", infer_env, args)
        infer_env.infer_saver = tf.train.Saver(max_to_keep=None)
        #infer_env.infer_saver = tf.train.Saver()

    return train_graph, infer_graph


def inferrence(args):
    args.load_model = True

    class Dummy:
        pass

    train_env = Dummy()
    infer_env = Dummy()
    _, infer_graph = model(args, train_env, infer_env)

    args.data = 'geosql'
    args.load_data = True
    X_train, y_train, X_test, y_test, X_dev, y_dev = load_data(args)
    model2load = 'model/{}'.format(args.subset)

    sess = tf.InteractiveSession(graph=infer_graph)
    infer_env.infer_saver.restore(sess, model2load)
    print('===========dev set============')
    decode_data(sess, infer_env, args, X_dev, y_dev)
    em = decode_data_recover(sess, infer_env, args, X_dev, y_dev, 'dev')
    print('==========test set===========')
    decode_data(sess, infer_env, args, X_test, y_test)
    test_em = decode_data_recover(sess, infer_env, args, X_test, y_test,
                                  'test')

    return


def train_model(args):
    class Dummy:
        pass

    train_env = Dummy()
    infer_env = Dummy()

    train_graph, infer_graph = model(args, train_env, infer_env)

    args.data = 'geo'
    args.load_data = True
    args.load_model = False
    X_train, y_train, X_test, y_test, X_dev, y_dev = load_data(args)
    model2load = 'model/{}'.format(args.subset)
    max_em, global_test_em, best_base = -1, -1, -1
    acc = 0
    sess1 = tf.InteractiveSession(graph=train_graph)
    sess1.run(tf.global_variables_initializer())
    sess1.run(tf.local_variables_initializer())
    sess2 = tf.InteractiveSession(graph=infer_graph)
    sess2.run(tf.global_variables_initializer())
    sess2.run(tf.global_variables_initializer())
    for base in range(args.total_epochs / args.epochs):
        print('\nIteration: %d (%d epochs)' % (base, args.epochs))
        model2load, acc = train(
            sess1,
            train_env,
            X_train,
            y_train,
            epochs=args.epochs,
            load=args.load_model,
            name=args.subset,
            batch_size=args.batch_size,
            base=base,
            acc=acc,
            model2load=model2load)
        args.load_model = True
        if acc > 0:
            infer_env.infer_saver.restore(sess2, model2load)

            print('===========dev set============')
            dev_em = decode_data(sess2, infer_env, args, X_dev, y_dev)
            #dev_em = decode_data_recover(sess, infer_env, args, X_dev, y_dev,
            #                             'dev')
            print('==========test set===========')
            test_em = decode_data(sess2, infer_env, args, X_test, y_test)
            #test_em = decode_data_recover(sess, infer_env, args, X_test, y_test,
            #                              'test')

            if dev_em > max_em:
                max_em = dev_em
                global_test_em = test_em
                best_base = base
                print('\n Saving model for best testing')
                train_env.saver.save(sess1, 'model/{0}-{1}-{2}'.format(args.subset, base, max_em))
            print('Max EM acc: %.4f during %d iteration.' % (max_em, best_base))
            print('test EM acc: %.4f ' % global_test_em)

    return



if __name__ == '__main__':
    args = init_args()
    print(args)
    if args.mode == 'train':
        print('\nTrain model.')
        train_model(args)
    elif args.mode == 'infer':
        print('\nInference.')
        inferrence(args)