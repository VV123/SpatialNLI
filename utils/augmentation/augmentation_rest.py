__author__ = 'Jingjing Li'

from StringIO import StringIO
import re
import pandas as pd
import random
import os

path = os.path.abspath(__file__)
PATH = os.path.dirname(path).replace('augmentation','annotate')
import sys
sys.path.insert(0, PATH)
from data_manager import read_word_rest

path = os.path.abspath(__file__)
save_path = os.path.dirname(path).replace('utils/augmentation',
                                          'data/DATA/rest/')
data_path = os.path.dirname(path) + '/data/rest/'
class Augmentation:
    def __init__(self):
        
        self.key_w, self.human_info, self.street, self.city, self.county, self.region, self.rest, self.foodtype, self.rating = read_word_rest()
        self.specific = self.street+self.city+self.county+self.region +self.rest + self.foodtype + self.rating
        with open(data_path + 'train_ground_truth.txt', 'r') as f1,\
            open(data_path+'train_org.qu','r') as f2:
            l1,l2 = f1.readlines(),f2.readlines()
            ans_org = []
            qu_org = []
            ans_aug = []
            qu_aug = []
            for line1,line2 in zip(l1,l2):
                line2 = line2.replace('?','')
                line2 = line2.strip('\n').split(' ')
                line2 = [item for item in filter(lambda x:x != '', line2)]
                line2 = ' '.join(line2)
                qu_aug.append(line2)

                line1 = line1.strip('\n').split(' ')
                line1 = [item for item in filter(lambda x:x != '', line1)]
                line1 = ' '.join(line1)
                ans_aug.append(line1)
            
            for line1,line2 in zip(l1,l2):
                line2 = line2.replace('?','')
                line2 = line2.strip('\n').split(' ')
                line2 = [item for item in filter(lambda x:x != '', line2)]
                line2 = ' '.join(line2)

                line1 = line1.strip('\n').split(' ')
                line1 = [item for item in filter(lambda x:x != '', line1)]
                line1_,line2_ = self.augmen(line1,line2)
                if line1_ != ' '.join(line1):
                    ans_aug.append(line1_)
                    qu_aug.append(line2_)

            with open(save_path + 'train_org.qu', 'w') as f:
                f.write('\n'.join(qu_aug))

            with open(save_path + 'train_ground_truth.txt', 'w') as f:
                f.write('\n'.join(ans_aug))


    
    def augmen(self,ans,qu):
        location = []
        dic = []
        ans_ = ' '.join(ans)
        for idx in range(len(ans)):
            if ans[idx] in ['cityid','streetid','countyid']:
                location.append([idx,ans[idx]])
        for loc,_ in location:
            for le in range(3, -1, -1):
                word = self._check_phrase(loc,le,ans)
                if word != None:
                    word = ' '.join(word)
                    dic.append(word)
                    break
        for idx,w in enumerate(dic):
            new_w = ''
            new_w_ = ''
            w = dic[idx]
            flag = location[idx][1]

            a = random.randint(0,100)

            if flag == 'cityid':
                a = self._check_random(a,w,self.city)
                new_w = self.city[a]
            elif flag == 'streetid':
                a = self._check_random(a,w,self.street)
                new_w = self.street[a]
            # elif flag == 'foodtypeid':
            #     a = self._check_random(a,w,self.foodtype)
            #     new_w = self.foodtype[a]
            elif flag == 'countyid':
                a = self._check_random(a,w,self.county)
                new_w = self.county[a]

            if new_w != '':
                ans_ = ans_.replace(w,new_w)
                qu = qu.replace(w,new_w)

        ans_ = ans_.strip('\n').split(' ')
        ans_ = [item for item in filter(lambda x:x != '', ans_)]
        ans_ = ' '.join(ans_)
        return ans_,qu
    
    def _check_random(self,a,w,words):
        while a >= len(words) or words[a] == w or '.' in words[a] or '#' in words[a]:
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