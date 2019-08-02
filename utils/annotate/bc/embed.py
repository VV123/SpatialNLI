import numpy as np
import os
path = os.path.abspath(__file__)
datapath = os.path.dirname(path).replace('/data','')
savepath = os.path.dirname(path).replace('/data','')
import codecs
import nltk
import glove
maxlen0 = 20
maxlen1 = 2

def embed_data(maxlen_p=maxlen0,maxlen_q=maxlen1,embedding=None,save=False):
	filepath ='wiki.npz'
	filepath_X = os.path.expanduser(os.path.join(savepath, filepath))
	filepath = 'wiki_label.npz'
	filepath_y = os.path.expanduser(os.path.join(savepath, filepath))
	g = embedding
	if g is None:
		g = glove.Glove()
	
	def _embedding(fpath):
		for line in codecs.open(fpath,'r','utf-8-sig'):
			# assert len(line.split('\t')) == 3 or line.startswith('#')
			assert len(line.split('\t')) == 3
		questions = [ nltk.word_tokenize(line.split('\t')[0]) for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		cols = [ nltk.word_tokenize(line.split('\t')[1]) for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		return g.embedding(questions, maxlen=maxlen_p-1), g.embedding(cols, maxlen=maxlen_q-1)


	def _read_label(fpath):
		#labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		labels = [ int(line.split('\t')[2].strip('\n')) for line in codecs.open(fpath,'r','utf-8-sig')]
		return labels

	print('\nGenerating training/test data')
	X_train_p,X_train_q = _embedding(os.path.join(datapath, 'train_model_const.txt'))
	X_test_p,X_test_q = _embedding(os.path.join(datapath, 'test_model_const.txt'))
	X_dev_p,X_dev_q = _embedding(os.path.join(datapath, 'dev_model_const.txt'))
	X_train_ans = _read_label(os.path.join(datapath, 'train_model_const.txt'))
	X_test_ans = _read_label(os.path.join(datapath, 'test_model_const.txt'))
	X_dev_ans = _read_label(os.path.join(datapath, 'dev_model_const.txt'))

	if save:
		print('\nSaving')
		np.savez(filepath_y, y_train=X_train_ans, y_test=X_test_ans, y_dev=X_dev_ans)
		np.savez(filepath_X, X_train_qu=X_train_p, X_train_col=X_train_q, X_test_qu=X_test_p, X_test_col=X_test_q, X_dev_qu=X_dev_p, X_dev_col=X_dev_q)
		print('\nSaved!')

def _embed_list(ls, g=None, maxlen_p=20, maxlen_q=2):
    if g == None:
        g = glove.Glove()
    for line in ls:
        assert len(line.split('\t')) == 2
    questions = [ nltk.word_tokenize(line.split('\t')[0]) for line in ls]
    cols = [ nltk.word_tokenize(line.split('\t')[1]) for line in ls]
    return g.embedding(questions, maxlen=maxlen_p-1), g.embedding(cols, maxlen=maxlen_q-1)

if __name__ == "__main__":
    #ls = ['give me the <f0> most <eof> sized city\tlargest', 'give me the <f0> most <eof> sized city\thighest']
    #X_inf_qu, X_inf_col = _embed_list(ls)
    embed_data(save = True)
