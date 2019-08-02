__author__ = 'Jingjing Li'

import os
import random
import numpy as np
import glove
import scipy.spatial.distance as dis
import nltk
import re
import pandas as pd 
from generate_lon import generate_lon
from data_manager import read_word_rest
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import bc.word_classifier_rest as wc
tf_model = wc.TF()

path = os.path.abspath(__file__)
PARSE_PATH = os.path.dirname(path).replace('utils/annotate',
                                          'data/DATA/rest/')
#PARSE_PATH = '/Users/lijing/Desktop/spatialNLI/data/rest'


class Question:

    phrase_length = 8 #max number of words in one phrase
    remove_stop = ['how','through','most','to','not','no','only','where','and','what']#the words need to be removed from the input stop words
    add_stop = ['would','find','called']#the words need to be added to the input stop words
    
    def __init__(self, parse=PARSE_PATH,split='train'):

        p_length = self.phrase_length 
        remove_stop = self.remove_stop
        add_stop = self.add_stop

        parse = os.path.expanduser(parse)
        rawfile='%s_org.qu'%split
        rawfile = os.path.join(parse, rawfile)

        self.split = split
        self.key_w, self.human_info, self.street, self.city, self.county, self.region, self.rest, self.foodtype, self.rating = read_word_rest()
        self.g = glove.Glove()
        self.embed()

        with open(rawfile, 'r') as f:

            res = []
            res_pair = []

            for i,line in enumerate(f):
                self.qu_pairs = []#words pairs

                self.count = 0 #number of <f>
                self.count_c = 0 #number of <c>

                words = word_tokenize(line)

                self.w_filter = words#line filtered out the stop words later
                self.qu_annot = ['']*len(words)#final result

                line = ' '.join(line.strip('\n').split(' '))

                self.stop_words = set(stopwords.words('english'))
                for rw in remove_stop:
                    self.stop_words.remove(rw)
                for aw in add_stop:
                    self.stop_words.add(aw)
                for k,w in enumerate(words):
                    if w not in self.stop_words:
                        self.w_filter[k] = w
                    else:
                        self.w_filter[k] = ''
                        if self.qu_annot[k] == '':
                            self.qu_annot[k]=w #append stop words in final result

                self.special_w_1(line)

                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.find_const_w(word, idx, le, line)

                #find the human knowledge word first before moving out stop words, since 'of' have to be moved out but then we can't find 'number of citizens'
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.find_human_w(word, idx, le, line)

                #the word exact match key word
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.exact_match(word, idx, le, line)

                self.special_w_2(line)

                if len(self.qu_annot) == len(self.w_filter):
                    for aw_idx, aw in enumerate(self.qu_annot):
                        if aw == '' and '<>' not in self.w_filter[aw_idx]:
                            self.qu_annot[aw_idx] = self.w_filter[aw_idx]
                else:
                    print('-------------wrong length for result--------------')
                    print(self.qu_annot)
                    print(self.w_filter)
                    print('----------------------------------------------------')

                qu_annot = [item for item in filter(lambda x:x != '', self.qu_annot)]
                qu_annot = ' '.join(qu_annot)
                qu_pairs = ''.join(self.qu_pairs)

                #all the key words have been picked up and labled, but their index number in <f+num> and <c+num> are not in order. Then we need to reorder the index. 
                qu_pairs, qu_annot = self.reorder(qu_pairs,qu_annot)
                
                res.append(qu_annot)
                res_pair.append(qu_pairs)

            print('\nSaving questions')
            with open(os.path.join(parse, '%s.qu'%self.split), 'w') as f:
                f.write('\n'.join(res))

            print('\nSaving pairs')
            with open(os.path.join(parse, '%s_sym_pairs.txt'%self.split), 'w') as f:
                f.write('\n'.join(res_pair))

        generate_lon(parse, split, 8)

        #restore the phrase(one or more words)
    def check_phrase(self,idx,le): 
        #'' should not be in a phrase ,'<' means this phrase has been annotated
        word = [] 
        for l in range(0,le+1):
            if self.w_filter[idx + l] == '' or '<' in self.w_filter[idx + l]:
                return
            word.append(self.w_filter[idx + l])

        return word
    def _lem(self, w):
        wnl = WordNetLemmatizer()
        lw = wnl.lemmatize(w)
        if lw == w and w.endswith('ing'):
            lw = wnl.lemmatize(w, pos='v')
        return lw

    def mark(self, flag, key, word, idx, typ = 'c'):
        word_ = ''
        if typ == 'insert':

            word_ += ' <f' +str(self.count) +'> '+ flag + ' <eof>' 

            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + flag +'<>'
            self.qu_pairs.append(pair)
            self.count += 1

            prefix = '<c' +str(self.count_c) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + word +'<>'#pairs for state name
            self.qu_pairs.append(pair)

            word_ +=' <c' +str(self.count_c) +'> '+ word + ' <eoc>'

            self.count_c += 1

            dic_pair = {'city': 'locations city name',
                   'rating': 'generalinfo rating',
                   'foodtype': 'generalinfo food type',
                   'restname': 'generalinfo rest name',
                   'street': 'locations street name',
                   'county': 'geographic county name',
                   'region': 'geographic region name'}

            for k, v in dic_pair.items():
                if key == k:
                    word_ +=' <f' +str(self.count) +'> '+ v + ' <eoc>'
                    prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
                    pair = prefix + '=>' + v +'<>'
                    self.qu_pairs.append(pair)
                    self.count += 1
            
            self.qu_annot[idx]=word_
            return 
        elif typ == 'c':
                word_ = '<c' +str(self.count_c) +'> '+ word + ' <eoc>'
                prefix = '<c' +str(self.count_c) +'>'
                pair = prefix + '=>' + flag +'<>'
                
                self.qu_pairs.append(pair)
                self.count_c += 1
                self.qu_annot[idx]=word_
        elif typ == 'f':
            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + flag +'<>'#pairs for state name
            word_ +='<f' +str(self.count) +'> '+ word + ' <eoc>'

            self.qu_pairs.append(pair)
            self.count += 1
            self.qu_annot[idx]=word_

    def mark_used(self,le,idx):
        for l in range(0,le+1):
            mark_w = '<>' + self.w_filter[idx + l] #for pharses have been mark by <f> or <c>, use <> to mark it
            self.w_filter[idx + l] = mark_w


    def find_human_w(self, word, idx, le, line):
        #singularize the word
        lm_word = word.split(' ')
        lm_word = ' '.join([self._lem(token) for token in lm_word])

        for key, pair in self.human_info.items():
            if word in pair or lm_word in pair:
                if lm_word in ['place','restaurant'] and 'where' in line:
                    pass
                elif lm_word in ['place','restaurant'] and 'how many' in line:
                    pass
                else:
                    self.mark_used(le, idx)
                    self.mark(key, None, word, idx,'f' )
                return
    
    def find_const_w(self, word, idx, le, line):
        if word in ['high point','chinese restaurant']:
            return
        if word == 'cafes':
            word = 'cafe'
        
        candidate = []
        dic_can = {'street': self.street,
                   'city': self.city,
                   'region': self.region,
                   'county': self.county,
                   'rest': self.rest,
                   'foodtype': self.foodtype,
                   'rating': self.rating}

        for k, v in dic_can.items():
            if word in v:
                candidate.append(k)

        if word in ['chinese','chinese food','cafe']:
            candidate = ['foodtype']
        
        if len(candidate) == 1:
            candidate = ''.join(candidate)
            if word == 'chinese food':
                word = 'chinese'
            if candidate == 'rest':
                candidate = 'restname'

            self.mark_used(le, idx)
            self.mark(candidate+'id', candidate, word, idx, 'insert')

        
        elif len(candidate) > 1:
            if word == 'chinese food':
                word = 'chinese'

            word_ = ''
            ls = []
            line_ =line.replace(word,'<f0> ' + word + ' <eof>')
            line_ = line_.strip('\n')
            line_ = line_.split(' ')
            line_ = ' '.join(line_)
            for w in ['street','city','region','rest','foodtype']:
                ls.append(line_ + '\t' + w)

            candidate, _ = tf_model.infer(ls, self.g)

            self.mark_used(le, idx)
            self.mark(candidate+'id', candidate, word, idx, 'insert')


    #find the word exactly same as key words in answer.
    def exact_match(self, word, idx, le, line):
        #singularize the word
        lm_word = word.split(' ')
        lm_word = ' '.join([self._lem(token) for token in lm_word])

        if lm_word in self.key_w or word in self.key_w:
            self.mark_used(le, idx)
            self.mark(lm_word, None, word, idx, 'f')

        else:
            porter_stemmer = PorterStemmer()
            for k_w in self.key_w:
                if porter_stemmer.stem(word) == porter_stemmer.stem(k_w):
                    self.mark_used(le, idx)
                    self.mark(k_w, None, word, idx, 'f')

    def embed(self): #all the key words embed from glove
        self.embed_key_w = []
        for d_w in self.key_w:
            embed = 0
            d_w_ = d_w.split(' ')
            for w in d_w_:
                embed += self.g.embed_one(w)
            self.embed_key_w.append([d_w, embed])

    def special_w_1(self, line):
        if 'what is' in line and 'bakery' in line:
            word_ = '<f' +str(self.count) +'> '+ 'what' + ' <eof>' 
            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + 'house no' +'<>'
            self.qu_pairs.append(pair)
            self.count += 1
            index = self.w_filter.index('what')
            self.qu_annot[index]=word_
            self.w_filter[index]='<>what'

    def special_w_2(self, line):
        if 'how many' in line:
            if 'foodtypeid' in ' '.join(self.qu_pairs):
                word_ = '<f' +str(self.count) +'> '+ 'how many' + ' <eof>' 
                prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
                pair = prefix + '=>' + 'count food type' +'<>'
                self.qu_pairs.append(pair)
                self.count += 1
                self.qu_annot[0]=word_
                self.qu_annot[1]=''
                self.w_filter[0]='<>how'
                self.w_filter[1]='<>many'
            if 'restnameid' in ' '.join(self.qu_pairs):
                word_ = '<f' +str(self.count) +'> '+ 'how many' + ' <eof>' 
                prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
                pair = prefix + '=>' + 'rest name count' +'<>'
                self.qu_pairs.append(pair)
                self.count += 1
                self.qu_annot[0]=word_
                self.qu_annot[1]=''
                self.w_filter[0]='<>how'
                self.w_filter[1]='<>many'
        if 'give me' in line and 'bakery' in line:
            word_ = '<f' +str(self.count) +'> '+ 'give me' + ' <eof>' 
            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + 'house no' +'<>'
            self.qu_pairs.append(pair)
            self.count += 1
            self.qu_annot[0]=word_
            self.qu_annot[1]=''
            self.w_filter[0]='<>give'
            self.w_filter[1]='<>me'
        if 'give me' in line and 'cafe' in line:
            word_ = '<f' +str(self.count) +'> '+ 'give me' + ' <eof>' 
            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + 'house no' +'<>'
            self.qu_pairs.append(pair)
            self.count += 1
            self.qu_annot[0]=word_
            self.qu_annot[1]=''
            self.w_filter[0]='<>give'
            self.w_filter[1]='<>me'

    def reorder(self, qu_pairs, qu_annot):
        for i in range(10):#to relocate <f+num> in sequence, we change '0' to '00','1' to '01'
            qu_pairs = qu_pairs.replace('f'+str(i)+'>','f0'+str(i)+'>')
            qu_annot = qu_annot.replace('f'+str(i)+'>','f0'+str(i)+'>')
            qu_pairs = qu_pairs.replace('c'+str(i)+'>','c0'+str(i)+'>')
            qu_annot = qu_annot.replace('c'+str(i)+'>','c0'+str(i)+'>')

        N_dic = {}
        for n,num in enumerate(qu_pairs):
            if num.isdigit() and qu_pairs[n+1] != '>':
                p = qu_pairs[n-1] + num +qu_pairs[n+1]
                N_dic[p] = n #the location for p is n, for example, p is f0, n is the location for 0 in self.qu_pairs

        loc = 0
        location = []
        location_c = []
        c_w_idx = [] # all of the <c+num>
        for loc,j in enumerate(qu_annot):
            if j == '<' and qu_annot[loc+1] == 'f':
                location.append([loc+2,loc+4])#the start location and end location of the phrase
            if j == '<' and qu_annot[loc+1] == 'c':
                location_c.append([loc+2,loc+4]) #the start location and end location of the phrase
                c_w_idx.append(qu_annot[loc+2:loc+4]) #<c+num>
        
        qu_annot_org = qu_annot
        qu_pairs = list(qu_pairs)
        new_qu_annot = list(qu_annot)
        f = 0 #the number that will be used to replace the numbers on the qu 
        
        for left,right in location:
            p_loc = str(qu_annot_org[left-1:right]) #the old <f+num>
            new_qu_annot[left] = list(str('%02d'%f))[0] #replace the old number by f, here f has two digits, '01' for '1'
            new_qu_annot[right-1] = list(str('%02d'%f))[1]
            p_idx = N_dic[p_loc] #get it index in qu_pairs
            qu_pairs[p_idx] = list(str('%02d'%f))[0]
            qu_pairs[p_idx+1] = list(str('%02d'%f))[1]
            f +=1
        f = 0

        for left,right in location_c:
            new_qu_annot[left] = list(str('%02d'%f))[0]
            new_qu_annot[right-1] = list(str('%02d'%f))[1]
            p_loc = str(qu_annot_org[left-1:right]) #the old <f+num>
            p_idx = N_dic[p_loc] #get it index in qu_pairs
            qu_pairs[p_idx] = list(str('%02d'%f))[0]
            qu_pairs[p_idx+1] = list(str('%02d'%f))[1]
            f += 1

        qu_annot = [item for item in filter(lambda x:x != '', qu_annot)]
        qu_annot = ''.join(new_qu_annot)
        qu_pairs = ''.join(qu_pairs)
        
        for i in range(10):#to relocate <f+num> in sequence, we change '0' to '00','1' to '01'
            qu_pairs = qu_pairs.replace('f0'+str(i)+'>','f'+str(i)+'>')
            qu_annot = qu_annot.replace('f0'+str(i)+'>','f'+str(i)+'>')
            qu_pairs = qu_pairs.replace('c0'+str(i)+'>','c'+str(i)+'>')
            qu_annot = qu_annot.replace('c0'+str(i)+'>','c'+str(i)+'>')
        qu_annot = qu_annot.replace('bos >','<bos>')

        return qu_pairs, qu_annot



if __name__ == '__main__':
    question = Question(split='train')
    question = Question(split='test')
    question = Question(split='dev')
