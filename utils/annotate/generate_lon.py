__author__ = 'Jingjing Li'

import os
import numpy as np
import pandas as pd
import json
from StringIO import StringIO
import re
import scipy.spatial.distance as dis

path = os.path.abspath(__file__)
PARSE_PATH = os.path.dirname(path).replace('utils/annotate',
                                          'data/DATA/geo/')
#PARSE_PATH = '/Users/lijing/Desktop/spatialNLI/data'

def generate_lon(parse=PARSE_PATH, split='train', phrase_length = 2):

    parse = os.path.expanduser(parse)
    ans_file ='%s_ground_truth.txt'%split
    pair_file ='%s_sym_pairs.txt'%split
    ans_file = os.path.join(parse, ans_file)
    pair_file = os.path.join(parse, pair_file)

    with open(ans_file) as f1,\
        open(pair_file) as f2:

        answer = []
        lines1, lines2 = f1.readlines(), f2.readlines()
        acc, n = 0, 0

        for ans, pair in zip(lines1, lines2):
    # pair = '<f0>=>area<><f1>=>stateid<><c0>=>alaska<>'
    # ans = '<f1> ( A B ( <f0> ( A ) loc ( B A ) <f2> ( B ) ) )'
    # qu = 'what is the <f0> area <eof> of <f1> stateid <eof> <c0> alaska <eoc>'

            pair_words = pair.split('<>')
            ans_words = ans.split(' ')
            pair_words = [item for item in filter(lambda x:x != '\n', pair_words)]
            pair_words = [item for item in filter(lambda x:x != '', pair_words)]
            
            for pair_w in pair_words:
                pair_w = pair_w.split('=>')
                
                for le in range(phrase_length, -1, -1):
                    for idx in range(len(ans_words)-le):
                        ans_w = ans_words[idx]

                        for l in range(1,le+1):
                            ans_w += ' ' + ans_words[idx + l]
                        if ans_w in ['biggest', 'highest','most','greatest','longest','maximum','largest','tallest']:
                            ans_w = 'argmax'
                        elif ans_w in ['fewest','smallest','minimum','sparsest','lowest','shortest','least']:
                            ans_w = 'argmin'


                        if ans_w == pair_w[1]:

                            ans_words[idx] = pair_w[0]
                            for l in range(1,le+1):
                                ans_words[idx + l] = ''
                            break

            ans_words = [item for item in filter(lambda x:x != '', ans_words)]
            ans_words = ' '.join(ans_words)


            answer.append(ans_words[:-1])

        print('\nSaving substitute')
        with open(os.path.join(parse, '%s.lon'%split), 'w') as f:
            f.write('\n'.join(answer))
