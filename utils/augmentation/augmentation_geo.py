__author__ = 'Jingjing Li'

from StringIO import StringIO
import re
import os
import pandas as pd
import random

path = os.path.abspath(__file__)
PATH = os.path.dirname(path).replace('augmentation','annotate')
import sys
sys.path.insert(0, PATH)
from data_manager import read_word_geo

path = os.path.abspath(__file__)
save_path = os.path.dirname(path).replace('utils/augmentation',
                                          'data/DATA/geo/')
data_path = os.path.dirname(path) + '/data/geo/'
class Augmentation:
    def __init__(self):
        
        self.const_w, self.key_w, self.human_info, self.special_w, self.city, self.state_name, self.state_abbre, self.state_place, self.country, self.lake, self.mountain, self.river = read_word_geo()
        self.specific = self.const_w + self.state_abbre
        print('--------------------finish load constant-----------------------')
        
        with open(data_path + 'train_org.qu', 'r') as f1,\
            open(data_path+'train_ground_truth.txt','r') as f2:
            
            l1,l2 = f1.readlines(),f2.readlines()
            
            self.qu = [] #question after deleting constant words
            self.qu_org = []#original input question

            self.ans = []#ground truth after deleting constant words
            self.ans_org = []#original input ground truth
            ans_ = [] #  in case of generating same sentence

            self.res_q = []#the output generated question after augmentation
            self.res_ans = []#the output generated ground truth after augmentation

            self.qu_shuf = [] #qu used for shuffling augmentation
            self.ans_shuf = []#ans used for shuffling augmentation

            self.letter_dict = [chr(i) for i in range(65,91)]

            #input original quetion
            for line1 in l1: #input original quetion
                line1 = line1.strip('\n').split(' ')
                line1 = [item for item in filter(lambda x:x != '', line1)]
                line1 = ' '.join(line1)
                self.qu_org.append(line1)
                self.res_q.append(line1)

            #input ground truth and substitute the constant phrase to get a new sentence
            for i,line2 in enumerate(l2):
                
                line2 = line2.strip('\n').split(' ')
                line2 = [item for item in filter(lambda x:x != '', line2)]
                
                self.res_ans.append(' '.join(line2))
                line2_, word = self._substitute(line2) #find out the constant word
                line2 = ' '.join(line2)
                self.ans_org.append(line2)
                
                if line2_ not in ans_:
                    ans_.append(line2_)#the new generated ground truth
                    if word != []:
                        self.ans.append(line2.replace(word,'<'+word+'>')) #The substituted word is marked by '<>'
                        self.qu.append(self.qu_org[i])#the new generated question

                if word != []:
                    self.ans_shuf.append(line2.replace(word,'<'+word+'>'))
                else:
                    self.ans_shuf.append(line2)
                self.qu_shuf.append(self.qu_org[i])
            print('--------------------finish load data-----------------------')


#replace the constant word: for example, 'What is the highest point in Florida'
# =====> 'What is the highest point in Alabama'

            self.replace_w()
            print('--------------------finish replace constant word-----------------------')


#concatenate two sentence: for example, 'What is the highest point in Florida' + 'what state has the 
#smallest population density' =====> 'What is the highest point in state that has the it smallest population density'
            for i,ans_1 in enumerate(self.ans):
                q_1 = self.qu[i]
                ans_1 = self.concatenate(ans_1,q_1,i,idx_ = 'state')
                ans_1 = self.concatenate(ans_1,q_1,i,idx_ = 'river')
                ans_1 = self.concatenate(ans_1,q_1,i,idx_ = 'city')
            print('--------------------finish concatenate sentences-----------------------')



#use 'and' to connect two sentence: for example, 'what state has the largest population' + 'what state has no rivers' 
#=====> 'what state has the largest population and has no rivers'
            for i,ans_1 in enumerate(self.ans_org):
                if i%4 == 0 : #don't need all original sentences to generate new one, 3 could be replaced, depend on how many sentences need to be generated.
                    q_1 = self.qu_org[i]
                    ans_1 = self.add_and(ans_1,q_1,i,idx_ = 'state')
            print('--------------------finish combine two sentences-----------------------')




#shuffling augmentation: for example, 'Which states does the Mississippi river run through?'
#=====> 'Through which states does the Mississippi river run'
            self.shuffling()
            print('--------------------finish shuffling augmentation-----------------------')



#maximum and minimum exchange: for example, 'what is the highest mountain in florida'
#=====> 'what is the minimum mountain in florida'
            self.max2min()
            print('--------------------finish max to min-----------------------')

            
            with open(save_path + 'train_org.qu', 'w') as f:
                f.write('\n'.join(self.res_q))
            with open(save_path + 'train_ground_truth.txt', 'w') as f:
                f.write('\n'.join(self.res_ans))

    def replace_w(self):
        for i, ans in enumerate(self.ans_org):
            qu = self.qu_org[i]

            location = []
            dic = []
            ans_ = ans
            for idx in range(len(ans)):
                if ans[idx] in ['cityid','stateid','riverid','placeid']:
                    location.append([idx,ans[idx]])
            
            for loc,_ in location:
                for le in range(3, -1, -1):
                    word = self._check_phrase(loc,le,ans)
                    if word != None:
                        word = ' '.join(word)
                        dic.append(word)
                        break
            
            if len(dic) == 1:
                new_w = ''
                new_w_ = ''
                w = dic[0]
                flag = location[0][1]

                a = random.randint(0,100)

                if flag == 'stateid':
                    a = self._check_random(a,w,self.state_name)
                    new_w = self.state_name[a]
                elif flag == 'riverid':
                    a = self._check_random(a,w,self.river)
                    new_w = self.river[a]
                elif flag == 'placeid':
                    a = self._check_random(a,w,self.state_place)
                    new_w = self.state_place[a]
                elif flag == 'cityid':
                    if len(w.split(' ')) == 1 or w.split(' ')[-1] not in self.state_abbre:
                        a = self._check_random(a,w,self.city)
                        new_w = self.city[a]
                    elif ' '.join(w.split(' ')[:-1]) in self.capital:
                        org_city = ' '.join(w.split(' ')[:-1])
                        org_state_abbre = w.split(' ')[-1]
                        if org_state_abbre == 'dc':
                            org_state = org_state_abbre
                        else:
                            org_state = self.state_name[self.state_abbre.index(w.split(' ')[-1])]
                        a = self._check_random(a,w,self.capital)
                        new_w_ = self.capital[a]
                        new_state = self.state_name[a]
                        new_state_abbre = self.state_abbre[a]

                if new_w != '':
                    ans_ = ans_.replace(w,new_w)
                    qu = qu.replace(w,new_w)
                elif new_w_ != '':
                    ans_ = ans_.replace(org_city,new_w_)
                    ans_ = ans_.replace(org_state_abbre,new_state_abbre)
                    qu = qu.replace(org_city,new_w_)
                    qu = qu.replace(org_state,new_state)

            ans_ = ans_.strip('\n').split(' ')
            ans_ = [item for item in filter(lambda x:x != '', ans_)]
            ans_ = ' '.join(ans_)
            
            if ans_ != ans and 'ga' not in self.qu_org[i]:
                self.res_ans.append(ans_)
                self.res_q.append(qu)

    def max2min(self):
        for i, ans in enumerate(self.ans_org):
            qu = self.qu_org[i]
            
            dic_max2min = {'largest':'smallest',
                           'longest':'shortest',
                           'most':'least'}
            
            for maxi, mini in dic_max2min.items():
                if maxi in qu:
                    ans, qu = self._exchange(maxi, mini, ans, qu)
                elif mini in qu:
                    ans, qu = self._exchange(mini, maxi, ans, qu)
            
            if 'highest' in qu:
                qu = qu.split(' ')
                if qu[qu.index('highest') + 1] != 'elevation':
                    qu = ' '.join(qu)
                    for w in ['maximum','minimum']:
                        ans_, qu_ = self._exchange('highest', w, ans, qu)
                else:
                    qu = ' '.join(qu)
                ans, qu = self._exchange('highest','lowest', ans, qu)
            
            elif 'lowest' in qu:
                qu = qu.split(' ')
                if qu[qu.index('lowest') + 1] != 'elevation':
                    qu = ' '.join(qu)
                    for w in ['maximum','minimum']:
                        ans_, qu_ = self._exchange('highest',w, ans, qu)
                else:
                    qu = ' '.join(qu)
                ans, qu = self._exchange('lowest','highest', ans, qu)

    def _exchange(self,w1,w2,line1,line2):
        if w1 in ['largest','longest','most','highest']:
            w1_ = 'argmax'
        elif w1 in ['smallest','shortest','least','lowest']:
            w1_ = 'argmin'
        
        if w2 in ['largest','longest','most','highest','maximum']:
            w2_ = 'argmax'
        elif w2 in ['smallest','shortest','least','lowest','minimum']:
            w2_ = 'argmin'
        
        if len([w for w in line1.split(' ') if w == w1_]) == 1:
            line2 = line2.replace(w1,w2)
            line1 = line1.replace(w1_,w2_)
            self.res_q.append(line2)
            self.res_ans.append(line1)
        # elif len([w for w in line1.split(' ') if w == w1_]) >1:
        #     print('')
        return line1,line2


    def shuffling(self):
        for i,ans_1 in enumerate(self.ans_shuf):
            q_1 = self.qu_shuf[i]
            if 'stateid' in ans_1:
                ans_1 = ans_1.strip('\n').split(' ')
                w1 = ''
                
                for k,w in enumerate(ans_1):
                    if '<' in w:
                        if '>' in w:
                            w1 = w.replace('<','').replace('>','')
                            break
                        else:
                            w1 += w.replace('<','')
                            w1 += ' '+ ans_1[k+1].replace('>','')
                
                ans_1 = ' '.join(ans_1)
                
                q_1 = q_1.split(' ')
                
                word = ''
                f = False
                for j in range(len(q_1)-1,0,-1):
                    word = q_1[j] + ' ' + word
                    if word[:-1] == w1:
                        f = True
                        break
                
                q_1 = ' '.join(q_1)
                if f == True and 'south dakota' not in q_1 and 'north dakota' not in q_1 and 'new jersey' not in q_1 and 'west virginia' not in q_1:
                    for letter in self.letter_dict:
                        if letter not in ans_1.replace(ans_1[ans_1.index('stateid')-10:ans_1.index('>')+5],''):
                            break
                    
                    letter = chr(ord(letter)-1)
                    for k,ans_2 in enumerate(self.ans_org):
                        q_2 = self.qu_org[k]
                        if 'river run through' in q_2 and 'which' in q_2:
                            q_2 = q_2.replace('which ','')
                            q_2 = q_2.replace(' through','')
                            q_2 = q_2.replace('does','through which')
                            q_1_ = q_1.replace(w1,q_2)

                            
                            if ans_2[0] == '(':
                                ans_2 = ans_2[1:-2]

                            ans_2 = self._replace_letter(ans_2, letter)
                            

                            ans_1_ = ans_1.replace(ans_1[ans_1.index('stateid')-11:ans_1.index(w1+'>')+len(w1)+5],ans_2)
                            if q_1_ not in self.res_q:
                                self.res_q.append(q_1_)
                                self.res_ans.append(ans_1_)
        
        for i,l in enumerate(self.qu_org):
            if 'in what state is' in l:
                self.res_q.append(l.split('is')[1] + ' is ' + l.split('is')[0])
                self.res_ans.append('( const' + self.ans_org[i][1:-1].split('const')[1] + self.ans_org[i][1:-1].split('const')[0]+')')


    def concatenate(self,ans_1,q_1,i,idx_):
        w_remove = ['new hampshire','red','called', 'colorado river','usa','towns','high point','alaska']#word won't be added in the generate sentence
        
        idx = idx_ +'id'

        if idx in ans_1 and len([w for w in w_remove if w in q_1]) == 0:#find state and pick up the name(const)
            
            ans_1 = ans_1.strip('\n').split(' ') #ans_1 is the first sentence
            
            w1 = '' # the 'state' in the first sentence
            for k,w in enumerate(ans_1):
                if '<' in w:
                    if '>' in w:
                        w1 = w.replace('<','').replace('>','')
                        break
                    else:
                        w1 += w.replace('<','')
                        w1 += ' '+ ans_1[k+1].replace('>','')
            
            if idx_ +' of '+w1 in q_1: #'the state of alabama' should be replaced together
                w1 = idx_ +' of '+w1
            if w1 + ' ' + idx_ in q_1: #'alabama state'
                w1 = w1 + ' ' + idx_
            if idx_ + ' '+ w1 in q_1: #'state alabama'
                w1 = idx_ + ' '+ w1
            if idx_ + 's '+ w1 in q_1: #'state alabama'
                w1 = idx_ + 's '+ w1
            if idx_ + ' called '+ w1 in q_1:#'city called new york'
                w1 = idx_ + ' called '+ w1
            if idx_ + ' named '+ w1 in q_1:#'city named new york'
                w1 = idx_ + ' named '+ w1
            if idx_ + 's named '+ w1 in q_1:#'rivers named colorado'
                w1 = idx_ + 's named '+ w1
            
            ans_1 = ' '.join(ans_1)
            q_1 = q_1.replace(' ?','') #q_1 is the first half sentence.

            for letter in self.letter_dict:
                if letter not in ans_1.replace(ans_1[ans_1.index(idx)-10:ans_1.index('>')+5],''): #pick up all the uppercase letter in the first sentence
                    break
            
            letter = chr(ord(letter)-1) # the start uppercase letter for the second sentence

            for j,ans_2 in enumerate(self.ans_org): #ans_2 is the second half sentence.

                
                if ans_2.split(' ')[0] =='(':#answer in the bracket, the bracket will be deleted
                    ans_2 = ans_2[2:-2]
                
                q_2 = self.qu_org[j] #q_2 is the second half sentence.
                q_2 = q_2.replace(' ?','')
                
                if 'what ' + idx_ in ' '.join(q_2.split(' ')[0:2]) and q_2 != q_1: #for those sentence describe the state and start with 'what', replace it to the constant in q_1
                   
                   q_1_, ans_1, ans_2_ = self._concat(q_2, q_1, w1, idx_, idx, ans_1, ans_2, letter, 'what')
                   self.res_q.append(q_1_) #add final generated concatenate question
                   self.res_ans.append(ans_1.replace(ans_1[ans_1.index(idx)-10:ans_1.index('>')+5],ans_2_))
                   
                elif 'which ' + idx_ in ' '.join(q_2.split(' ')[0:2]) and q_2 != q_1: #for those sentence describe the state and start with 'which', replace it to the constant in q_1

                    q_1_, ans_1, ans_2_ = self._concat(q_2, q_1, w1, idx_, idx, ans_1, ans_2, letter, 'which')

                    if q_1_ not in self.res_q:
                        self.res_q.append(q_1_)
                        self.res_ans.append(ans_1.replace(ans_1[ans_1.index(idx)-10:ans_1.index('>')+5],ans_2_))
                
        return ans_1

    def add_and(self, ans_1, q_1, i, idx_):
        idx = idx_ +'id'
        if 'what ' + idx_  in ' '.join(q_1.split(' ')[0:2]) and 'usa' not in ans_1:#find state and pick up the name(const)
            q_1 = q_1.replace(' ?','')

            for letter in self.letter_dict:
                if letter not in ans_1:
                    break
            letter = chr(ord(letter)-1)

            for j,ans_2 in enumerate(self.ans_org):
                
                if ans_2.split(' ')[0] =='(':#answer in the bracket, the bracket will be deleted
                    ans_2 = ans_2[2:-2]
                
                q_2 = self.qu_org[j]
                q_2 = q_2.replace(' ?','')
                
                if j%3 == 0: #don't need all original sentences to generate new one, 3 could be replaced, depend on how many sentences need to be generated.
                    if 'what ' + idx_ in ' '.join(q_2.split(' ')[0:2]) and q_2 != q_1 and 'state' in ans_2.split(' ')[0] and 'usa' not in ans_2: #for those sentence describe the state, replace it to the constant in q_1
                        
                        q_2 = self._replace(q_2, idx_, 'what', 'and')

                        q_1_ = q_1+' '+q_2

                        self.res_q.append(q_1_)
#'A' will be replaced by 'C', but then 'C' will continue to be replaced, so use @ to mark that as being replaced
                        ans_2_ = self._replace_letter(ans_2, letter)

                        ans_2_ = ans_2[:-2] + ans_2_[11:] +' ) )'
                        self.res_ans.append(ans_2_)
        return ans_1

    def _replace(self, q_2, idx_, start_w, middle_w):

        if idx_ == 'state':
            q_2 = q_2.replace(start_w + ' states ', middle_w + ' ')
        elif idx_ == 'city':
            q_2 = q_2.replace(start_w + ' cities ', middle_w + ' ')
        elif idx_ == 'river':
            q_2 = q_2.replace(start_w + ' rivers ', middle_w + ' ')
        q_2 = q_2.replace(start_w + ' ' + idx_ + ' ', middle_w + ' ')
        q_2 = q_2.replace('does','')
        
        return q_2

    
    def _replace_letter(self, ans_2, letter):
        ans_2_ = ans_2
        for letter_1 in self.letter_dict:
            if letter_1 not in ans_2:
                pass
            if letter_1 in ans_2:
#'A' will be replaced by 'C', but then 'C' will continue to be replaced, so use @ to mark that as being replaced
                ans_2_ = ans_2_.replace(' '+letter_1,' @'+letter)
                letter = chr(ord(letter)+1)
        ans_2_ = ans_2_.replace('@','')
        return ans_2_


    def _concat(self, q_2, q_1, w1, idx_, idx, ans_1, ans_2, letter, start_w):
        
        q_2 = self._replace(q_2, idx_, start_w, 'that')

        q_1_ = q_1.replace(w1,idx_ + ' '+q_2)

        ans_1_ = ans_1
        for le in self.letter_dict:
            if le in ans_1_:
                ans_1_ = ans_1_.replace(le, '')
        ans_1_ = ans_1_.replace('(','').replace(')','')
        ans_1_ = ans_1_.split(' ')
        ans_1_ = [item for item in filter(lambda x:x != '', ans_1_)]

        if ans_1_.index(idx)+2 < len(ans_1_) and ans_1_[ans_1_.index(idx)+2] == idx_:
            ans_1 = ans_1.replace(ans_1[ans_1.index(idx_ + ' '):ans_1.index(idx_ + ' ')+len(idx_ + ' ')],'') # delete the specifi 'state' name in the first sentence
        #the word 'state ( B )' in the original lambda should be replaced
        if idx_ ==  ans_1_[ans_1_.index(idx)-2]:
            ans_1 = ans_1.replace(ans_1[ans_1.index(idx)-(18+len(idx_)):ans_1.index(idx)-11],'')

        ans_2_ = self._replace_letter(ans_2, letter)

        return q_1_, ans_1, ans_2_

    def _substitute(self,line):
        phrase_length = 3
        flag = False
        line_ = ' '.join(line)
        word_ = ''
        for le in range(phrase_length - 1, -1, -1):
            for idx in range(len(line)-le):
                word = self.check_phrase(idx,le,line)
                if word != None:
                    word = ' '.join(word)
                    if word in self.specific:
                        flag = True
                        line_ = line_.replace(word,'')
                        word_ = word
        line_ = line_.strip('\n').split(' ')
        line_ = [item for item in filter(lambda x:x != '', line_)]
        if flag == True:
            return line_,word_
        return [],[]

    def check_phrase(self,idx,le,line1): 
        #'' should not be in a phrase ,'<' means this phrase has been annotated
        word = [] 
        for l in range(0,le+1):
            if line1[idx + l] == '' or '<' in line1[idx + l]:
                return
            word.append(line1[idx + l])

        return word

    def _check_random(self,a,w,words):
        while a >= len(words) or words[a] == w or words[a] == 'district of columbia':
            a = random.randint(0,100)
        return a


    def _check_phrase(self,idx,le,line1): 
        word = [] 
        for l in range(2,le+1):
            if line1[idx + l] == ')':
                break
            word.append(line1[idx + l])

        return word

if __name__ == '__main__':
    augmentation = Augmentation()