__author__ = 'Jingjing Li'
import os

data_path = ''
save_path = 'geo'
with open(os.path.join(data_path, 'dev_infer.txt'), 'r') as f1,\ # test file (actual inferred)
    open(os.path.join(data_path, 'dev_org.qu'),'r') as f2,\
    open(os.path.join(data_path, 'train_ground_truth.txt'),'r') as f3,\ # train file
    open(os.path.join(data_path, 'train_org.qu'),'r') as f4,\ 
    open(os.path.join(data_path, 'test_ground_truth.txt'), 'r') as f5,\ # test file (gold)
    open(os.path.join(data_path, 'test_org.qu'),'r') as f6:
    lines1, lines2, lines3, lines4, lines5, lines6 = f1.readlines(), f2.readlines(),f3.readlines(),f4.readlines(),f5.readlines(),f6.readlines()
    acc, n = 0, 0
    answer = []
    dev = []
    train = []
    test = []
    for l1, l2, l5,l6 in zip(lines1, lines2,lines5,lines6):
        l1 = l1.strip('\n')
        l2 = l2.strip('\n')
        l5 = l5.strip('\n')
        l6 = l6.strip('\n')
        l1_ = 'answer ( A ' + l1 + ' )'
        l5_ = 'answer ( A ' + l5 + ' )'
        l1_ = l1_.replace('next to','next_to')
        l5_ = l5_.replace('next to','next_to')

        l1_ = l1_.split(' ')
        l1 = []
        for i,w in enumerate(l1_):
            if w not in ['A','B','C','D','E','F','G','H','I','(',')'] and 'const' not in l1_[i-5]:
                if 'const' in l1_[i-6] and l1_[i-1] != '(' and l1_[i+1] == ')':
                    if len(w) == 2:
                        w_ = ''
                        w_ = ', ' +w
                        w = w_
                    else:
                        w = w + " '"
                        line = "' "+l1[i-1]
                        l1[i-1] = line
                else:
                    w = '_'+w
            if w ==')' and i-3 < len(l1_) and l1_[i-3] == 'cityid' and len(l1_[i-1]) != 2:
                w = ', _ )'
            elif w ==')' and i-4 < len(l1_) and l1_[i-4] == 'cityid' and ')' not in l1[i-1] and len(l1_[i-1]) != 2:
                w = ', _ )'
            if w in ['A','B','C','D','E','F','G','H','I',')']:
                if i+1 < len(l1_) and l1_[i+1] != ')':
                    w = w + ' ,'
            l1.append(w)
        l1 = ' '.join(l1)
        l1 = l1.replace('_not','\+')
        l1 = l1.replace('argmax','largest')
        l1 = l1.replace('argmin','smallest')
        l1 = l1.replace('_high _point','_high_point')
        l2 = l2.replace('argmax','largest')
        l2 = l2.replace('argmin','smallest')
        dev.append(l2 + '\t' + l1)

        l5_ = l5_.split(' ')
        l5 = []
        for i,w in enumerate(l5_):
            if w not in ['A','B','C','D','E','F','G','H','I','(',')'] and 'const' not in l5_[i-5]:
                if 'const' in l5_[i-6] and l5_[i-1] != '(' and l5_[i+1] == ')':
                    if len(w) == 2:
                        w_ = ''
                        w_ = ', ' +w
                        w = w_
                    else:
                        w = w + " '"
                        line = "' "+l5[i-1]
                        l5[i-1] = line
                else:
                    w = '_'+w
            if w ==')' and i-3 < len(l5_) and l5_[i-3] == 'cityid' and len(l5_[i-1]) != 2:
                w = ', _ )'
            elif w ==')' and i-4 < len(l5_) and l5_[i-4] == 'cityid' and ')' not in l5[i-1] and len(l5_[i-1]) != 2:
                w = ', _ )'
            if w in ['A','B','C','D','E','F','G','H','I',')']:
                if i+1 < len(l5_) and l5_[i+1] != ')':
                    w = w + ' ,'
            l5.append(w)
        l5 = ' '.join(l5)
        l5 = l5.replace('_not','\+')
        l5 = l5.replace('argmax','largest')
        l5 = l5.replace('argmin','smallest')
        l5 = l5.replace('_high _point','_high_point')
        l6 = l6.replace('argmax','largest')
        l6 = l6.replace('argmin','smallest')
        test.append(l6 + '\t' + l5)
        
    flag = 0
    for l3, l4 in zip(lines3, lines4):
        if flag <600:
            l3 = l3.strip('\n')
            l4 = l4.strip('\n')
            l3_ = 'answer ( A ' + l3 + ' )'
            l3_ = l3_.replace('next to','next_to')
            l3_ = l3_.split(' ')
            l3 = []
            for i,w in enumerate(l3_):
                if w not in ['A','B','C','D','E','F','G','H','I','(',')'] and 'const' not in l3_[i-5]:
                    if 'const' in l3_[i-6] and l3_[i-1] != '(' and l3_[i+1] == ')':
                        if len(w) == 2:
                            w_ = ''
                            w_ = ', ' + w
                            w = w_
                        else:
                            w = w + " '"
                            line = "' "+l3[i-1]
                            l3[i-1] = line
                    else:
                        w = '_'+w
                if w ==')' and i-3 < len(l3_) and l3_[i-3] == 'cityid' and len(l3_[i-1]) != 2:
                    w = ', _ )'
                elif w ==')' and i-4 < len(l3_) and l3_[i-4] == 'cityid' and ')' not in l3[i-1] and len(l3_[i-1]) != 2:
                    w = ', _ )'
                if w in ['A','B','C','D','E','F','G','H','I',')']:
                    if i+1 < len(l3_) and l3_[i+1] != ')':
                        w = w + ' ,'
                l3.append(w)
            l3 = ' '.join(l3)
            l3 = l3.replace('_not','\+')
            l3 = l3.replace('argmax','largest')
            l3 = l3.replace('argmin','smallest')
            l3 = l3.replace('_high _point','_high_point')
            l4 = l4.replace('argmax','largest')
            l4 = l4.replace('argmin','smallest')
            train.append(l4 +' ?' '\t' + l3)
        flag += 1


print('\nSaving txt doc')
with open(os.path.join(save_path, 'train.tsv'), 'w') as f:
    f.write('\n'.join(train))
with open(os.path.join(save_path, 'dev.tsv'), 'w') as f:
    f.write('\n'.join(dev))
with open(os.path.join(save_path, 'test.tsv'), 'w') as f:
    f.write('\n'.join(test))
