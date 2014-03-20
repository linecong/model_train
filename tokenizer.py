#coding=utf-8
from chr_def import *
import os
import mmseg
import codecs
from jianfan import ftoj

def tradition_to_simple(str):
	return ftoj(str)

def normalize_str(str):
	str=tradition_to_simple(str)
	chr_list=[]
	for w in str:
		if w.encode('utf-8') in punct_chr_set:
			pass
		elif is_digit(w):
			chr_list.append('0')
		else:
			if isinstance(w,unicode):
				w=w.encode('utf-8')
				chr_list.append(w)
	return ''.join(chr_list)

def extract_grams(str,n):
	if not isinstance(str,unicode):
		str=str.decode('utf-8')
	fts_list=[]
	strlen=len(str)
	i=0
	while i < strlen:
		if is_ascii_char(str[i]):
			ascii_list=[]
			while i < strlen:
				if is_ascii_char(str[i]):
					ascii_list.append(str[i])
				else:
					break
				i+=1
			if len(ascii_list) > 0:
				ascii_ft=''.join(ascii_list)
				fts_list.append(ascii_ft.encode('utf-8'))
		if i + n - 1 < strlen:
			chr_list=[]
			for d in range(n):
				if is_digit(str[i+d]):
					chr_list.append('0')
				elif is_letter(str[i+d]):
					chr_list.append('A')
				elif is_ascii_char(str[i+d]):
					chr_list.append('_')
				else:
					chr_list.append(str[i+d])
			fts_list.append("".join(chr_list).encode('utf-8'))
		i+=1
	return fts_list

def extract_trigrams(str):
	return extract_grams(str,3)

def extract_bigrams(str):
	return extract_grams(str,2)

def extract_unigrams(str):
	return extract_grams(str,1)

class Segmentor(object):
	def __init__(self):
		mmseg.mmseg.dict_load_words(os.path.join(os.path.dirname(__file__), 'dict', 'words.dic'))
		mmseg.mmseg.dict_load_chars(os.path.join(os.path.dirname(__file__), 'dict', 'chars.dic'))
		stopwords_path = os.path.join(os.path.dirname(__file__), 'dict', 'stopwords.txt')
		self.stopwords = map(lambda x: x.strip().encode('utf-8'), tuple(codecs.open(stopwords_path, 'r', 'utf-8')))
		replace_words_path = os.path.join(os.path.dirname(__file__), 'dict', 'words.replace')
		self.replace_dict={}
		self.max_len_to_replace=0
		for line in open(replace_words_path):
			elems=line.strip().split('\t')
			self.replace_dict[elems[0]] = elems[1]
			word_len=len(elems[0])
			if word_len > self.max_len_to_replace:
				self.max_len_to_replace = word_len

	def normalize_syn_words(self,text):
		tokens=mmseg.seg_txt(text)
		word_list=[x for x in tokens]
		wlist_len=len(word_list)
		for i in xrange(wlist_len):
			if word_list[i] == "":
				continue
			curr_len=0
			j = i
			while j < wlist_len:
				curr_len+=len(word_list[j])
				if curr_len > self.max_len_to_replace:
					break
				j+=1
			while j > i:
				wrf="".join(word_list[i:j])
				if wrf in self.replace_dict:
					wrt=self.replace_dict[wrf]
					word_list[i] = wrt
					for k in xrange(i+1,j):
						word_list[k] = ""
					break
				j-=1
		return "".join(word_list)

	def remove_stop_words(self,text):
		tokens=mmseg.seg_txt(text)
		left_words=[]
		for t in tokens:
			if t not in self.stopwords:
				left_words.append(t)
		return "".join(left_words)

	def segword(self, text):
		tokens = mmseg.seg_txt(text)
		return [x for x in tokens]

if __name__ == "__main__":
	segm=Segmentor()
	#line="买衣服,啥时候可以到货啊?你们是从哪儿发货"
	#line="没了　啊"
	#line="老客户了，给个折扣嘛 "
	line="我要確認收穫先嗎"
	str=normalize_str(line)
	print str
	tokens=segm.segword(line)
	for t in tokens:
		print t,
	print
	print segm.remove_stop_words(line)
	print segm.normalize_syn_words(line)
	print extract_bigrams(str)
