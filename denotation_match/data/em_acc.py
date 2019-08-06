dataset = 'geo_0.9'
with open('{0}/dev.tsv'.format(dataset), 'r') as f1,\
	open('{0}/test.tsv'.format(dataset),'r') as f2,\
	open('{0}/train.tsv'.format(dataset),'r') as f3:
	lines1, lines2, lines3 = f1.readlines(), f2.readlines(), f3.readlines()
	acc, n = 0, 0
	answer = []
	res = []
	for l1,l2 in zip(lines1,lines2):
		if l1.split() == l2.split():
			acc += 1
		else:
			print(l1)
			print(l2)
		n += 1
	print(acc*1.0/n)
print(acc)

