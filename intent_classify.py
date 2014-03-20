#coding=utf-8
import sys
import codecs
import linecache
import logging
import os
import cPickle
import math
from numpy import random
import numpy as np
from scipy import stats
from gensim import matutils, corpora, models
from gensim.corpora import TextCorpus
from gensim.models import TfidfModel
from scipy.sparse import dok_matrix
from sklearn import covariance, svm
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.multiclass import *
from sklearn.naive_bayes import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def build_train_dict(name, ft_set, persist=True):
    if os.path.exists('imodel/dict.q'):
        dictionary = corpora.Dictionary.load('imodel/dict.q')
        logging.info("load dict from file.")
    else:
        texts = []
        for line in open(name):
            elems=line.strip().split('\t')
            if len(elems) >= 2:
                texts.append(elems[1].strip().split())
            else:
                texts.append([])
                print "nil=%s" % line
        numdoc=len(texts)
        dictionary = corpora.Dictionary(texts)
        #once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if (docfreq <= 3 or docfreq*1.0/numdoc>1.0)] 
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() \
		if dictionary.__getitem__(tokenid) not in ft_set] 
        dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
        dictionary.compactify()
        logging.info("num of key=%d" % (len(dictionary.keys())))
        if persist:
            dictionary.save('imodel/dict.q')
    return dictionary

class FeatureSpace(object):
	def __init__(self, train_data_path):
		self.ft_set = self.get_ft_set("ft.txt")
		self.train_dict = build_train_dict(train_data_path,self.ft_set)
		self.train_data_path = train_data_path

	def get_ft_set(self, path):
		ft_set=set({})
		for line in open(path):
			elems = line.strip().split('\t')
			ft_set.add(elems[0])
		return ft_set

	def get_feature_vector(self,tokens):
		vec=self.train_dict.doc2bow(tokens)
		n_dim = len(self.train_dict.keys())
		x=[0] * n_dim
		for i,value in vec:
			x[i] = value
		return x
	
	def get_train_input_matrix(self):
		points = []
		for line in open(self.train_data_path):
			elems=line.strip().split('\t')
			if len(elems) >= 2:
				tokens = elems[1].split()
			else:
				tokens = []
			fts=self.get_feature_vector(tokens)
			points.append(fts)
		return np.array(points)

def get_label_vector(path):
   Y=[]
   for line in open(path):
	  label=line.strip().split('\t')[0]
	  y = int(label)
	  Y.append(y)
   return Y

def get_predict_class_and_prob(clf,X):
	y = clf.predict(X)
	#probs = clf.predict_proba(X)
	return y,1.0

class PrecStat(object):
	def __init__(self):
		self.cls_right_num = {}
		self.cls_pred_num={}
		self.cls_tag_num={}

	def add_instance(self, pred_label, tag_label):
		if pred_label == tag_label:
			if pred_label not in self.cls_right_num:
				self.cls_right_num[pred_label] = 0
			self.cls_right_num[pred_label] += 1
		if tag_label not in self.cls_tag_num:
			self.cls_tag_num[tag_label] = 0
		self.cls_tag_num[tag_label] += 1
		if pred_label not in self.cls_pred_num:
			self.cls_pred_num[pred_label] = 0
		self.cls_pred_num[pred_label] += 1
	
	def output(self):
		for i in self.cls_tag_num:
			tmp_right_num=0
			if i in self.cls_right_num:
				tmp_right_num=self.cls_right_num[i]
			recall=tmp_right_num * 1.0 / self.cls_tag_num[i]
			if i in self.cls_pred_num:
				prec=tmp_right_num * 1.0 / self.cls_pred_num[i]
			else:
				prec=0.0
			if prec < 0.6 or recall < 0.6:
				print "%d: prec=%f recall=%f *" % (i, prec,recall)
			else:
				print "%d: prec=%f recall=%f" % (i, prec,recall)
		tot_right_num=sum(self.cls_right_num.values())
		tot_pred_num=sum(self.cls_pred_num.values())
		#print "tot_right_num=%d" % tot_right_num
		#print "tot_pred_num=%d" % tot_pred_num
		tot_prec=tot_right_num * 1.0 / tot_pred_num
		print "tot prec=%f" % tot_prec
		useful_right_num=tot_right_num-self.cls_right_num[0]
		useful_pred_num=tot_pred_num-self.cls_pred_num[0]
		useful_tag_num=sum(self.cls_tag_num.values())-self.cls_tag_num[0]
		print "useful prec=%f" % (useful_right_num * 1.0 / useful_pred_num)
		print "useful recall=%f" % (useful_right_num * 1.0 / useful_tag_num)

def test_model(test_data_path):
	prec_stat=PrecStat()
	for line in open(test_data_path):
		elems=line.strip().split('\t')
		test_label=int(elems[0])
		if len(elems) >= 2:
			tokens=elems[1].split()
		else:
			tokens=[]
		X_test=feature_space.get_feature_vector(tokens)
		y,p=get_predict_class_and_prob(clf,X_test)
		prec_stat.add_instance(y[0],test_label)
		print line.strip(), y, p
	prec_stat.output()

if __name__ == '__main__':
	train_data_path=sys.argv[1]
	test_data_path=sys.argv[2]
	feature_space = FeatureSpace(train_data_path)
	
	if os.path.exists('imodel/qcls.model'):
		with open('imodel/qcls.model', 'rb') as fid:
			clf=cPickle.load(fid)
			logging.info("load question classify model from file.")
	else:
		X=feature_space.get_train_input_matrix()
		Y=get_label_vector(train_data_path)
		#clf=RandomForestClassifier(n_estimators=30)
		#clf=ExtraTreesClassifier(n_estimators=30)
		clf=svm.LinearSVC(C=0.15,tol=1e-7)
		clf.fit(X,Y)
		with open('imodel/qcls.model', 'wb') as fid:
			cPickle.dump(clf, fid)

	test_model(test_data_path)
