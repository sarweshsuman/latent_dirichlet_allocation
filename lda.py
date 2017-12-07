import os
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
from gensim import corpora

def read_data(location):
	files=os.listdir(location)
	documents=[]
	for fn in files:
		full_pth = os.path.join(location,fn)
		fh = open(full_pth,'r')
		lines=fh.readlines()
		fh.close()
		documents.extend(lines)
	return documents

def preprocess_documents(documents):
	stopwrds = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	punct = set(string.punctuation)
	new_documents=[]
	for doc in documents:
		lower_doc = doc.lower()
		words = word_tokenize(lower_doc)
		n_words = [lemmatizer.lemmatize(w) for w in words if w not in stopwrds]
		new_doc = ''.join([ch for ch in " ".join(n_words) if ch not in punct])
		new_documents.append(word_tokenize(new_doc)) # corpora.Dictionary takes documents as collection of tokens
	return new_documents

def create_document_term_matrix(documents):
	dictionary = corpora.Dictionary(documents)
	document_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
	return (dictionary,document_term_matrix)

def process(dictionary,document_term_matrix):
	lda = gensim.models.ldamodel.LdaModel
	ldamodel= lda(document_term_matrix,num_topics=5,id2word = dictionary,passes=100)
	print(ldamodel.print_topics(num_topics=5,num_words=10))

if __name__ == '__main__':
	documents=read_data('./data')
	documents=preprocess_documents(documents)
	(dictionary,document_term_matrix ) = create_document_term_matrix(documents)
	process(dictionary,document_term_matrix)
