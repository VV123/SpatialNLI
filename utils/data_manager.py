# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#sys.path.append('/Users/lijing/Desktop/NLIDB-master/utils')
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glove
from collections import defaultdict
import argparse
import vocab
from vocab import build_vocab_all, load_vocab_all

_PAD = vocab._PAD
_GO = vocab._GO
_END = vocab._END
_EOF = vocab._EOF
_UNK = vocab._UNK
_TOKEN_NUMBER = vocab._TOKEN_NUMBER
_TOKEN_MODEL = vocab._TOKEN_MODEL
_EOC = vocab._EOC


def _load_data(args, s='train',d='geo'):
    dir_path = args.data_path
    geo_path = dir_path + '/DATA/%s'%(d)
    save_path = dir_path
    maxlen = args.maxlen
    load = args.load_data
    if load:
        emb = np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        print('========embedding shape========')
        print(emb.shape)
        all_q_tokens = np.load(os.path.join(save_path,'embed/%s_qu_idx.npy'%(s)))
        all_logic_ids = np.load(os.path.join(save_path,'embed/%s_lon_idx.npy'%(s)))
        print('all_q_tokens.shape:', all_q_tokens.shape)
        print('all_logic_ids.shape:', all_logic_ids.shape)
    else:
        all_q_tokens = []
        all_logic_ids = []
        vocab_dict,_,_,_=load_vocab_all(args, d)
        vocab_dict = defaultdict(lambda:_UNK, vocab_dict)
        questionFile=os.path.join(geo_path,'%s.qu'%(s))
        logicFile=os.path.join(geo_path,'%s.lon'%(s))
        with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
            q_sentences = questions.readlines()
            logics = logics.readlines()
            assert len(q_sentences)==len(logics)
            i = 0
            length = len(logics)
            for q_sentence,logic in zip(q_sentences,logics):
                i+=1
                #print('counting: %d / %d'%(i,length),end='\r')
                sys.stdout.flush()
                token_ids = [_GO]
                token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
                for x in q_sentence.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in question:'+x)
                #token_ids.append(_END)
                logic_ids = [_GO]
                logic_ids.extend([vocab_dict[x] for x in logic.split()])
                for x in logic.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in logic:'+x)
                logic_ids.append(_END)
                if maxlen>len(logic_ids):
                    logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
                else:
                    logic_ids = logic_ids[:maxlen]
                if maxlen>len(token_ids):
                    token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
                else:
                    token_ids = token_ids[:maxlen]
                all_q_tokens.append(token_ids)
                all_logic_ids.append(logic_ids)
            all_logic_ids=np.asarray(all_logic_ids)
            #print('------wiki '+s+' shape------')
            #print(all_logic_ids.shape)
            all_q_tokens=np.asarray(all_q_tokens)
            np.save(os.path.join(save_path,'embed/%s_lon_idx.npy'%s),all_logic_ids)
            np.save(os.path.join(save_path,'embed/%s_qu_idx.npy'%s),all_q_tokens)
    
    return all_q_tokens,all_logic_ids


def load_data(args):
    maxlen = args.maxlen
    load = args.load_data
    if args.data == 'geo':
        X_train, y_train = _load_data(args,s='train',d='geo')
        print('========Train data shape=======')
        print(X_train.shape)
        print(y_train.shape) 
        X_test, y_test = _load_data(args,s='test',d='geo')
        print('========Test data shape=======')
        print(X_test.shape)
        print(y_test.shape)
        X_dev, y_dev = _load_data(args,s='dev',d='geo')
        print('========Dev data shape=======')
        print(X_dev.shape)
        print(y_dev.shape) 
        return X_train, y_train, X_test, y_test, X_dev, y_dev
    elif args.data == 'rest':
        X_train, y_train = _load_data(args,s='train',d='rest')
        print('========Train data shape=======')
        print(X_train.shape)
        print(y_train.shape) 
        X_test, y_test = _load_data(args,s='test',d='rest')
        print('========Test data shape=======')
        print(X_test.shape)
        print(y_test.shape)
        X_dev, y_dev = _load_data(args,s='dev',d='rest')
        print('========Dev data shape=======')
        print(X_dev.shape)
        print(y_dev.shape) 
        return X_train, y_train, X_test, y_test, X_dev, y_dev


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--maxlen', default=60, type=int, help='Data record max length.')  
    arg_parser.add_argument('--data', default='geo', type=str, help='Data.')
    arg_parser.add_argument('--embedding_dim', default=300, type=int, help='Embedding dim.')
    arg_parser.add_argument('--data_path', default=os.path.dirname(os.path.abspath(__file__)).replace('utils', 'data'), type=str, help='Data path.')
    arg_parser.add_argument('--load_data', default=False, type=bool, help='Load data.')
    args = arg_parser.parse_args()

    rebuild = True   
    if rebuild:
        build_vocab_all(args, load=False,d=args.data)
        load_vocab_all(args, load=False,d=args.data)
        load_data(args)


