from StringIO import StringIO
import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

path = os.path.abspath(__file__)
PATH = os.path.dirname(path).replace('/bc','')

import sys
sys.path.insert(0, PATH)
from data_manager import read_word_geo,read_word_rest

DATA_PATH_GEO = os.path.dirname(path) + '/data/geo/'
DATA_PATH_REST = os.path.dirname(path)+ '/data/rest/'

def find_const_w_geo(word, idx, le, city, state_name, state_place, country, lake, mountain, river):
    candidate = []
    dic_can = {'country': country,
               'mountain': mountain,
               'lake': lake,
               'place': state_place,
               'city': city,
               'river': river,
               'state': state_name}

    for key, v in dic_can.items():
        if word in v:
            candidate.append(key)
    
    return candidate

def find_const_w_rest(word, idx, le, street, city, region, county, rest, foodtype, rating):
    candidate = []
    dic_can = {'street': street,
               'city': city,
               'region': region,
               'county': county,
               'rest': rest,
               'foodtype': foodtype,
               'rating': rating}

    for k, v in dic_can.items():
        if word in v:
            candidate.append(k)
    
    return candidate

#restore the phrase(one or more words)
def check_phrase(idx,le, qu): 
    #'' should not be in a phrase ,'<' means this phrase has been annotated
    word = [] 
    for l in range(0,le+1):
        word.append(qu[idx + l])

    return word


def bc(parse = DATA_PATH_GEO, split = 'train', data = 'geo'):

    remove_stop = ['how','through','most','to','not','no']
    p_length = 0 # max number of words in one phrase
    dim = 25

    if data == 'geo':
        const_w, key_w, human_info, special_w, city, state_name, state_abbre, state_place, country, lake, mountain, river = read_word_geo()
        p_length = 3
        word_list = ['city','state','river'] #potential label for ambiguous phrase
    else:
        key_w, human_info, street, city, county, region, rest, foodtype, rating = read_word_rest()
        p_length = 8
        word_list = ['street','city','region','rest','foodtype']
    
    qu_file='%s_org.qu'%split
    qu_file = os.path.join(parse, qu_file)

    ans_file='%s_ground_truth.txt'%split
    ans_file = os.path.join(parse, ans_file)
    
    if split == 'train':
        times = len(word_list)-1 #repeat times of the '1' in the training set for one ambiguous phrase
    else:
        times = 1 #for test, don't need to repeat '1'

    res = []

    with open(qu_file,'r') as f1,\
        open(ans_file,'r') as f2:
        l1, l2 = f1.readlines(), f2.readlines()
        for qu,ans in zip(l1,l2):
            
            qu = qu.replace(',',' ')
            qu = qu.replace('?','')
            qu = qu.replace('.', '')
            qu = qu.replace("'", '')
            qu_w = qu

            count = 0
            words = qu.strip('\n').split(' ')

            stop_words = set(stopwords.words('english'))
            for rw in remove_stop:
                stop_words.remove(rw)

            w_filter = words
            
            for k,w in enumerate(words):
                if w not in stop_words:
                    w_filter[k] = w
                else:
                    w_filter[k] = ''

            qu_w = w_filter
            for le in range(p_length-1, -1, -1):
                for idx in range(len(qu_w)-le):
                    
                    word = check_phrase(idx,le, qu_w) 
                    if word != None:
                        word = ' '.join(word)
                        if data == 'geo':
                            candidate = find_const_w_geo(word, idx, le, city, state_name, state_place, country, lake, mountain, river)#pick up the phrase
                        else:
                            candidate = find_const_w_rest(word, idx, le, street, city, region, county, rest, foodtype, rating)
                        if 'place' in candidate:
                            candidate.remove('place')
                        if len(candidate) > 1:#the candidate phrase for spatial comprehension model(has two potential label)
                            
                            for c in candidate:
                                if c + 'id' in ans:
                                    
                                    qu_ = qu.replace(word,'<f0> '+ word +' <eof>')
                                    qu_ = qu_.strip('\n').split(' ')
                                    
                                    for i in range(len(qu_),dim):
                                        qu_.append('<pad>')
                                    
                                    qu_ = ' '.join(qu_)
                                    for j in range(0,times):
                                        line1 = qu_ + '\t' + c+ '\t' + '1'
                                        res.append(line1)

                                    word_list.remove(c)
                                    for j in word_list:
                                        line2 = qu_ + '\t' + j+ '\t' + '0'
                                        res.append(line2)
                                    word_list.append(c)
                                    
                                    break

    print('\nSaving bc_data')
    with open(os.path.join(parse, '%s_model_const.txt'%split), 'w') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    data = 'rest' #'geo' dataset or 'rest' datasetDATA_PATH_GEO = os.path.dirname(path).replace('utils/bc',
    if data == 'geo':
        bc(split = 'train')
        bc(split = 'test')
        bc(split = 'dev')
    else:
        bc(parse = DATA_PATH_REST, split = 'train', data = 'rest')
        bc(parse = DATA_PATH_REST, split = 'test', data = 'rest')
        bc(parse = DATA_PATH_REST, split = 'dev', data = 'rest')