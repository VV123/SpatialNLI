res = []
# with open('dev_ground_truth.txt', 'r') as f1,\
# 	open('dev_infer.txt','r') as f2,\
dataset = 'geo_0.9'
with open('{0}/dev.tsv'.format(dataset), 'r') as f1,\
	open('{0}/test.tsv'.format(dataset),'r') as f2,\
	open('{0}/train.tsv'.format(dataset),'r') as f3:
	lines1, lines2, lines3 = f1.readlines(), f2.readlines(), f3.readlines()
	acc, n = 0, 0
	answer = []
	res = []
	for l1,l2 in zip(lines1,lines2):
	# for l2 in lines2:
	# 	l2 = l2.strip('\n')
	# 	answer.append(l2)
	# for l3 in lines3:
	# 	l3 = l3.strip('\n')
	# 	answer.append(l3)
	# for l1 in lines1:
	# 	l1 = l1.strip('\n')
	# 	if l1 not in answer:
	# 		if 'argmax' not in l1 and 'argmin' not in l1:
	# 			print(l1)
	# 			acc += 1
		if l1.split() == l2.split():
			acc += 1
		else:
			print(l1)
			print(l2)
			# res.append(str(n))
			# res.append(' '.join(l1.split()))
			# res.append(' '.join(l1_.split()))
			# res.append(' '.join(l2.split()))
		n += 1
	print(acc*1.0/n)
print(acc)

# print('\nSaving txt doc')
# with open('wrong_line.txt', 'w') as f:
#     f.write('\n'.join(res))
