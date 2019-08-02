__author__ = 'Jingjing Li'

import os
import numpy as np
import glove
import scipy.spatial.distance as dis
import nltk
import re
import pandas as pd 
import string
from StringIO import StringIO
from generate_lon import generate_lon
from data_manager import read_word_geo
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import bc.word_classifier as wc
tf_model = wc.TF()

path = os.path.abspath(__file__)
DATA_PATH = os.path.dirname(path).replace('utils/annotate',
                                          'data/DATA/geo/')
#DATA_PATH = '/Users/lijing/Desktop/spatialNLI/data/geo'



class Question:

    phrase_length = 3 #max number of words in one phrase
    remove_stop = ['how','through','most','to','not','no'] #the words need to be removed from the input stop words

    def __init__(self, data_path=DATA_PATH,split='train'):

        p_length = self.phrase_length 
        remove_stop = self.remove_stop

        data_path = os.path.expanduser(data_path)
        rawfile='%s_org.qu'%split
        rawfile = os.path.join(data_path, rawfile)
        
        self.split = split
        self.const_w, self.key_w, self.human_info, self.special_w, self.city, self.state_name, self.state_abbre, self.state_place, self.country, self.lake, self.mountain, self.river = read_word_geo()
        #self.read_word()
        self.g = glove.Glove()
        self.embed()
        
        with open(rawfile, 'r') as f:

            res = []
            res_pair = []

            for i,line in enumerate(f):
                
                self.qu_pairs = []#word pairs

                self.count = 0 #index of <f>
                self.count_c = 0 #index of <c>
                
                line = line.replace('?', '')
                line = line.replace(',', ' ')
                line = line.replace('.', '')
                line = line.replace("'", '')

                words = word_tokenize(line)

                self.w_filter = words#line filtered out the stop words
                self.qu_annot = ['']*len(words)#final result

                line = ' '.join(line.strip('\n').split(' '))

                self.find_loc(line) #annotate the words corresponding to 'loc' in lambda(most of them are in stop words)
        
        #find the human knowledge word first before moving out stop words, since 'of' have to be moved out but then we can't find 'number of citizens'
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.find_human_w(word, idx, le, line)

                self.stop_words = set(stopwords.words('english'))
                for rw in remove_stop:
                    self.stop_words.remove(rw) #remove stop words

                for k,w in enumerate(words):
                    if w not in self.stop_words:
                        self.w_filter[k] = w
                    else:
                        self.w_filter[k] = ''
                        if self.qu_annot[k] == '':
                            self.qu_annot[k]=w #append stop words in final result (these stop words will not be labled)

                #find the constant word <c>
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.find_const_w(word, idx, le, line)

                #the word exact match key word
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.exact_match(word, idx, le, line)
                
                #exact first then compute the cos distance
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.w_filter)-le):
                        word = self.check_phrase(idx,le)
                        if word != None:
                            word = ' '.join(word)
                            self.cos_distance(word, idx, le)

                qu_annot = [item for item in filter(lambda x:x != '', self.qu_annot)]
                qu_annot = ' '.join(qu_annot) #final result
                qu_pairs = ''.join(self.qu_pairs) #final word pairs

                #all the key words have been picked up and labled, but their index number in <f+num> and <c+num> are not in order. Then we need to reorder the index. 
                qu_pairs, qu_annot = self.reorder(qu_pairs,qu_annot)
                
                res.append(qu_annot)
                res_pair.append(qu_pairs)

            print('\nSaving questions')
            with open(os.path.join(data_path, '%s.qu'%self.split), 'w') as f:
                f.write('\n'.join(res))

            print('\nSaving pairs')
            with open(os.path.join(data_path, '%s_sym_pairs.txt'%self.split), 'w') as f:
                f.write('\n'.join(res_pair))
        
        generate_lon(data_path, split, 2)

    #restore the phrase(one or more words)
    def check_phrase(self,idx,le): 
        #'' should not be in a phrase ,'<' means this phrase has been annotated
        word = [] 
        for l in range(0,le+1):
            if self.w_filter[idx + l] == '' or '<' in self.w_filter[idx + l]:
                return
            word.append(self.w_filter[idx + l])

        return word


    def embed(self): #all the key words embed from glove
        self.embed_key_w = []
        for d_w in self.key_w:
            embed = 0
            d_w_ = d_w.split(' ')
            for w in d_w_:
                embed += self.g.embed_one(w)
            self.embed_key_w.append([d_w, embed])


    def _lem(self, w):
        wnl = WordNetLemmatizer()
        lw = wnl.lemmatize(w)
        if lw == w and w.endswith('ing'):
            lw = wnl.lemmatize(w, pos='v')
        return lw

    def mark(self, flag, key, word, idx, typ = 'c'):
        word_ = ''
        if typ == 'insert':
            word_ = '<f' +str(self.count) +'> '+ flag + ' <eof>' 

            prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + flag +'<>'
            self.qu_pairs.append(pair)

            prefix = '<c' +str(self.count_c) +'>'#get the prefix<> for pairs
            pair = prefix + '=>' + key +'<>'#pairs for state name
            self.qu_pairs.append(pair)
            word_ +=' <c' +str(self.count_c) +'> '+ word + ' <eoc>'

           #self.count_c = self.count + 1
            self.count_c += 1
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
            word_ +='<f' +str(self.count) +'> '+ word + ' <eof>'

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

                self.mark_used(le, idx)
                self.mark(key, None, word, idx,'f' )
                return

    def find_const_w(self, word, idx, le, line):
        if word == 'high point':
            return

        dic_can = {'country': self.country,
                   'mountain': self.mountain,
                   'lake': self.lake,
                   'place': self.state_place,
                   'city': self.city,
                   'river': self.river,
                   'state': self.state_name,
                   'abbre': self.state_abbre}
        
        candidate = []

        for key, v in dic_can.items():
            if word in v:
                candidate.append(key)

        if len(candidate) == 1 or len(candidate) == 2 and len([w for w in ['place', 'lake'] if w in candidate])>0:
            #print(candidate,'1111111')

            if len(candidate) == 1:
                candidate = ''.join(candidate)
            else:
                for w in ['place','lake']:
                    if w in candidate:
                        candidate.remove(w)
                candidate = ''.join(candidate)

            self.mark_used(le, idx)#use '<>' to mark the phrase(one or more words) that has been annotated

            if candidate == 'state':#for annotate the abbrevation state after the city, for example, auburn, AL
                if 'cityid' in ' '.join(self.qu_annot) or self.w_filter[idx-1] in self.city:

                    abbre = self.state_abbre[self.state_name.index(word)]
                    self.mark(abbre, None, word, idx, 'c')

                    return

            if candidate == 'abbre': #for abbre, no need to insert
                self.mark(word, None, word, idx, 'c')
                return
            if candidate == 'country': 
                self.mark(candidate+'id', 'usa', word, idx, 'insert')
                return

            self.mark(candidate+'id', word, word, idx, 'insert')

        elif len(candidate) > 1:
            
            self.mark_used(le, idx)#use '<>' to mark the phrase(one or more words) that has been annotated

            if 'state' in candidate:
                if 'cityid' in ' '.join(self.qu_annot) or self.w_filter[idx-1] in self.city:
                    abbre = self.state_abbre[self.state_name.index(word)]
                    self.mark(abbre, None, word, idx, 'c')
                    return

            if 'city' in candidate and 'state' in candidate and idx+1 <len(self.w_filter):
                if self.w_filter[idx+1] in self.state_abbre or self.w_filter[idx+1] in self.state_abbre:
                    flag = 'city'
                    self.mark(flag+'id', word, word, idx, 'insert' )
                    return
            if word == 'washington':
                if idx+1 >= len(self.w_filter) or idx+1 < len(self.w_filter) and self.w_filter[idx+1] not in self.state_abbre:
                    flag = 'state'
                    self.mark(flag+'id', word, word, idx, 'insert' )
                    return


            ls = []
            line_ =line.replace(word,'<f0> ' + word + ' <eof>')
            line_ = line_.strip('\n')
            line_ = line_.split(' ')
            line_ = ' '.join(line_)
            for w in ['city','state','river']:
                ls.append(line_ + '\t' + w)

            flag, prob = tf_model.infer(ls, self.g) #use spatial model to pick up its label

            self.mark(flag+'id', word, word, idx, 'insert' )


    #find the word exactly same as key words in answer.
    def exact_match(self, word, idx, le, line):
        candidate = ''
        #singularize the word
        lm_word = word.split(' ')
        max_w = ['biggest', 'highest','most','greatest','longest','maximum','largest','tallest']
        min_w = ['fewest','smallest','minimum','sparsest','lowest','shortest','least']
        lm_word = ' '.join([self._lem(token) for token in lm_word])
         #to identify if the word is in the speical word list, then annotate it as the corresponding word
        for key, pair in self.special_w.items():
            if word in pair or lm_word in pair:
                if key == 'usa':
                    candidate = 'country'
                elif word == 'elevation' and self.w_filter[idx-1] not in ['<>highest','<>lowest','<>maximum','maximum','highest','lowest']:
                    pass
                else:
                    self.mark_used(le, idx)
                    self.mark(key,  None, word, idx, 'f')
                    return
        if word == 'population' and idx+1 < len(self.w_filter) and self.w_filter[idx+1] in ['density','densities']:
            pass

        elif candidate != '':
            self.mark_used(le, idx)

        #for annotate the abbrevation state after the city, for example, auburn, AL
            if candidate == 'state' and 'cityid' in ' '.join(self.qu_annot):
                abbre = self.state_abbre[self.state_name.index(self.w_filter[idx].replace('<>',''))]
                self.mark(abbre, None, word, idx, 'c')
                return

            if candidate == 'abbre': #for abbre, no need to insert
                self.mark(word, None, word, idx, 'c')
                return
            if candidate == 'country': 
                self.mark(candidate+'id', 'usa', word, idx, 'insert')
                return

            self.mark(candidate+'id', word, word, idx, 'insert')

        elif word in max_w:
            flag = 'argmax'

            self.mark_used(le, idx)
            self.mark(flag, None, word, idx, 'f')
        elif word in min_w:
            flag = 'argmin'

            self.mark_used(le, idx)
            self.mark(flag, None, word, idx, 'f')
        
        elif lm_word in self.key_w or word in self.key_w:
            if lm_word == 'border':
                lm_word = 'next to'

            self.mark_used(le, idx)
            self.mark(lm_word, None, word, idx, 'f')

        else:
            porter_stemmer = PorterStemmer()
            for k_w in self.key_w:
                if porter_stemmer.stem(word) == porter_stemmer.stem(k_w):
                    
                    self.mark_used(le, idx)
                    self.mark(k_w, None, word, idx, 'f')
    
    def cos_distance(self, word, idx, le):
        words = word.split(' ')
        w_embed = 0
        for w in words:
            w_embed += self.g.embed_one(w)

        cos = 1
        w = ''
        for d_w,emb in self.embed_key_w:
            cos_ = dis.cosine(emb, w_embed)
            if cos_ < cos:
                cos = cos_
                w = d_w
        if le > 0:
            threshold = 0.25
        else:
            threshold = 0.25

        if cos < threshold:
            if word == 'population' and self.w_filter[idx+1].replace('<>','') == 'density':
                pass
            elif word == 'go':
                pass
            else:
                self.mark_used(le, idx)
                word = '<f' +str(self.count) +'> '+ word + ' <eof>'

                prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
                pair = prefix + '=>' + w +'<>'
                self.qu_pairs.append(pair)


                self.count += 1

        self.qu_annot[idx]=word 

    def find_loc(self,line):
        self.loc_word = ['states','all states','all the states','the largest state','the largest city','the state','rivers','a city','the states','the lowest point','the highest point','cities','a state','a river','continantal us','the largest state','the highest elevation','the lowest elevation','the longest river','the smallest state','the most major cities','those','the city','the biggest city','the most rivers','a major river','the most cities','fewest cities','most rivers','the largest capital city','each state']
        loc_dic = []
        loc_idx = []
        for idx,word in enumerate(self.w_filter):
            if word in ['in','of','with','has','have','contains','on','where','whose']:
                loc_dic.append(word)
                loc_idx.append(idx)
        
        wrong_w = 0
        for w in ['how many citizens','how many inhabitants','how many residents','how many people','how many population','square kilometers']:
            if w in line: 
                wrong_w += 1

        if len(loc_dic) > 0 and wrong_w == 0:
            for i,word in enumerate(loc_dic):
                loc_flag = self._find_loc(word,loc_idx[i])
                if loc_flag == True:
                    self.w_filter[loc_idx[i]] = '<>' + word # use '<>' to mark this word showing it has been annotated
                    self.mark('loc',  None, word, loc_idx[i], 'f')

        elif len(loc_dic) > 1:
            count_loc = 0
            for i,word in enumerate(loc_dic):
                loc_flag = self._find_loc(word,loc_idx[i]) 
                if loc_flag == True:
                    count_loc += 1
            if count_loc > 1:
                self.w_filter[loc_idx[i]] = '<>' + word
                self.mark('loc',  None, loc_dic[1], loc_idx[i], 'f')
        
        if 'how many people live in the capital of' in line:
            self.mark('loc',  None, 'of', self.w_filter.index('of'), 'f')
            self.w_filter[self.w_filter.index('of')] = '<>' + word

    def _loc_check_phrase(self,word,idx,start_loc,loc_w):
        length = min(len(self.w_filter)-idx-1,5)
        for le in range(length, -1, -1): #see if the phrases after this word in the const, if it is, the word is paired with 'loc'
            if idx+1 < len(self.w_filter):
                pos_w = []
                for l in range(start_loc,le+1):
                    pos_w.append(self.w_filter[idx + l])
                if pos_w != []:
                    pos_w = ' '.join(pos_w)
                    if pos_w in self.const_w:
                        return True
                    elif loc_w == True and pos_w in self.loc_word:
                        return True
        return False
    
    def _loc_check_phrase_left(self,word,idx,start_loc):
        for le in range(4, 1, -1):
            pre_w = [] 
            for l in range(idx-le,idx-start_loc):
                pre_w.append(self.w_filter[l])

            if pre_w != None:
                pre_w = ' '.join(pre_w)
                if pre_w in self.const_w and 'population' not in self.w_filter and self.w_filter[idx-1] not in ['city']:
                    return True
        return False
    
    def _find_loc(self,word,idx):
        if word == 'where':
            if ' '.join(self.w_filter[idx+1:idx+3]) == 'are mountains':
                return True
            elif ' '.join(self.w_filter[idx+1:idx+5]) == 'is the smallest city':
                return True
            elif self.w_filter[idx+1] in ['are','is']:
                return self._loc_check_phrase(word,idx,2,False)
        
        if word == 'on' and self.w_filter[idx-1] != 'lie':
            return True
        if word == 'whose':
            return True
        if word == 'of' and self.w_filter[idx-1] == 'number':
            return False
        if word in ['has','have']:
            if idx + 2 < len(self.w_filter) and self.w_filter[idx+1] == 'the' and self.w_filter[idx+2] == 'city':
                return True
            if 'citizens' in self.w_filter:
                return False
            if idx + 1 < len(self.w_filter) and self.w_filter[idx+1] == 'no':
                return False
            if len([w for w in ['population','populations','area','density','people'] if w in self.w_filter]) == 0:
                return True
            else:
                return False
        if word in ['in','of'] and self.w_filter[idx-1] in ['population','populations','people','area','density','largest','length','name','height','elevation','state','which']:
            return False
        if idx+1 < len(self.w_filter) and self.w_filter[idx+1] in ['meters']:
            return False
        if word == 'contains':
            return True
        
        if self. _loc_check_phrase(word,idx,1,True):
            return True
        if idx+1 < len(self.w_filter) and self.w_filter[idx+1] == 'capital' :
            if self. _loc_check_phrase(word,idx,2,False):
                return True
        if idx+1 < len(self.w_filter) and ' '.join(self.w_filter[idx+1:idx+3]) == 'the capital': 
            if self. _loc_check_phrase(word,idx,3,False):
                return True
        
        if self._loc_check_phrase_left(word,idx,0):
            return True
        
        if self.w_filter[idx-1] == 'located' :
            if self._loc_check_phrase_left(word,idx,1):
                return True

        if ' '.join(self.w_filter[idx-3:idx]) == 'is the capital':
            if self._loc_check_phrase_left(word,idx,3):
                return True
        if self.w_filter[idx-1] == 'capital' and self.w_filter[idx-2] == 'the':
            if self._loc_check_phrase_left(word,idx,2):
                return True
        return False

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
        #c_w_idx_dic = {}
        
        for left,right in location:
            p_loc = str(qu_annot_org[left-1:right]) #the old <f+num>
            #if qu_annot[left:right] in c_w_idx: #for <c>, make sure it has the same index with the <f> before it, for example, <f3> stateid <c3> virginia.
                #c_w_idx_dic[qu_annot[left:right]] = f
            new_qu_annot[left] = list(str('%02d'%f))[0] #replace the old number by f, here f has two digits, '01' for '1'
            new_qu_annot[right-1] = list(str('%02d'%f))[1]
            p_idx = N_dic[p_loc] #get it index in qu_pairs
            qu_pairs[p_idx] = list(str('%02d'%f))[0]
            qu_pairs[p_idx+1] = list(str('%02d'%f))[1]
            f +=1
        
        f = 0

        for left,right in location_c:
            #if qu_annot[left:right] in list(c_w_idx_dic.keys()): #get the index number for <c>, it should be same as the <f> before that.
                #f = c_w_idx_dic[qu_annot[left:right]]
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
