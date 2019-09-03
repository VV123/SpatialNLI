import re
with open('geo880_test280.tsv', 'r') as f:
    parse_answer = []
    question = []
    for i,line in enumerate(f):
        print(line)
        print(i)
        line = line.strip('\n').split('\t')
        # m = re.match(r'^(.*)*_answer\(A,(.*\))\).*$', line)
        # m = re.match(r'^.*parse\(\[(.*)\], *(answer\(A,.*\))\).*$', line)
        question.append(line[0])
        ans = line[1].replace('_','')
        ans = ans.replace('nextto','next to')
        ans = ans.replace('answer ( A ,','')
        ans = ans.replace('\+','not')
        ans = ans.replace('highpoint','high point')
        ans = ans.replace('lowpoint','low point')
        ans = ans.replace(',','')
        ans = ans.replace("'",'')
        ans = ans[:-1]
        ans = ans.strip('\n').split(' ')
        ans = [item for item in filter(lambda x:x != '', ans)]
        ans = ' '.join(ans)
        parse_answer.append(ans)


print('\nSaving questions')
with open('test_org.qu', 'w') as f:
    f.write('\n'.join(question))

print('\nSaving answers')
with open('test_ground_truth.txt', 'w') as f:
    f.write('\n'.join(parse_answer))
