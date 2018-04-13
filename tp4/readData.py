#!usr/bin/env python3
#pylint:disable-all

import sys
# from random import randint
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

f=open("BaseReuters-29")

all_lab=[]
row=[]
column=[]
freq=[]
doc_size=0

# pour chaque ligne du fichier
for line in f:
	doc_size=doc_size+1		#on incremente la taille du fichier
	l=line.split()			#on coupe a chaque espace de chaque ligne

	all_lab.append(int(l[0]))	#le premier caractere est ajoute, c'est le numero de la ligne
	for i in range(1,len(l)):
		feat=l[i].split(":")	#feat est un tableau contenant deux nombres
		row.append(doc_size-1)	#dans row on ajoute le numero de la ligne
		column.append(int(feat[0]))		#on ajoute dans column le premier element de feat
		freq.append(int(feat[1]))		#on ajoute dans freq le deuxieme element de feat

vocab_size=np.max(column)	#vocab_size est le maximum de column
max_class=np.max(row)		#max_class est le max de row
print(vocab_size)		#on affiche vocab_size

# all_lab.append(1)

#on stocke dans une matrice: (numero du document, numero du terme) frequence d'apparition du terme dans ce document
reuters=sparse.csr_matrix((freq,(row,column)),shape=(doc_size,vocab_size+1))

# -------------------------------- Overall Aim --------------------------------

# init tab taille 29
doc_per_class = [0 for _ in range(0, 29)]

# fill the tab to get the number of documents for each class
for element in all_lab:
	doc_per_class[element-1] += 1

# Question 2)
#print(doc_per_class)

# ----------------- Bernoulli vs Multonomial distributions ------------------

#  1)
training_set, test_set, training_lab, test_lab = train_test_split(reuters, all_lab, train_size = 52500 , test_size = 18203)

# initialise the model
bernoulli_model = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

# we train the supervised model
bernoulli_model.fit(training_set, training_lab)

#  2)
# initialise the model
multinomial_model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# we train the supervised model
multinomial_model.fit(training_set, training_lab)

#  3)
predict_bernoulli = bernoulli_model.predict(test_set)
predict_multinomial = multinomial_model.predict(test_set)

# estimation of the accuracy of bernoulli_model
misclassified_rate_bernoulli = 0
for index, elt_bernoulli in enumerate(predict_bernoulli):
	if elt_bernoulli != test_lab[index]:
		misclassified_rate_bernoulli += 1
misclassified_rate_bernoulli /= len(test_lab)
print("Rate of errors using the Bernoulli model : ", misclassified_rate_bernoulli*100)

# estimation of the accuracy of multinomial_model
misclassified_rate_multinomial = 0
for index, elt_multinomial in enumerate(predict_multinomial):
	if elt_multinomial != test_lab[index]:
		misclassified_rate_multinomial += 1
misclassified_rate_multinomial /= len(test_lab)
print("Rate of errors using the Multinomial model : ", misclassified_rate_multinomial*100)


#  4) Cross-validation 

training_set1, test_set1, training_lab1, test_lab1 = train_test_split(reuters, all_lab, train_size = 56562 , test_size = 14141)
training_set2, test_set2, training_lab2, test_lab2 = train_test_split(training_set1, training_lab1, train_size = 42421 , test_size = 14141)
training_set3, test_set3, training_lab3, test_lab3 = train_test_split(training_set2, training_lab2, train_size = 28280 , test_size = 14141)
training_set4, test_set4, training_lab4, test_lab4 = train_test_split(training_set3, training_lab3, train_size = 14140 , test_size = 14140)

train_sets=[]	#faut trouver une structure de données pour y stocker training_set1, training_set2, training_set3 et training_set4
train_labs=[]	#faut trouver une structure de données pour y stocker ici training_lab1, training_lab2, training_lab3 et training_lab4
test_sets=[]	#faut trouver une structure de données pour y stocker ici test_set1, test_set2, test_set3 et test_set4


# Bernoulli_model

for i in range(5):
	bernoulli_model = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
	bernoulli_model.fit(train_sets[i], train_labs[i])
	predict_bernoulli = bernoulli_model.predict(test_sets[i])


# Multinomial_model

for i in range(5):
    multinomial_model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    multinomial_model.fit(train_sets[i], train_labs[i])
    predict_multinomial = multinomial_model.predict(test_sets[i])
