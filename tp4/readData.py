#!usr/bin/env python3
#pylint:disable-all


# Poupeau Alexandre
# Julien Eloise
# Bouzbiba Lamyaa

# ------------- Document Classification with Generative Naive-Bayes Models --------------


import sys
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

f=open("BaseReuters-29")

all_lab=[]
row=[]
column=[]
freq=[]
doc_size=0

# pour chaque ligne du fichier
for line in f:
	doc_size=doc_size+1		# on incremente la taille du fichier
	l=line.split()			# on coupe a chaque espace de chaque ligne

	all_lab.append(int(l[0]))	# le premier caractere est ajoute, c'est le numero de la ligne
	for i in range(1,len(l)):
		feat=l[i].split(":")	# feat est un tableau contenant deux nombres
		row.append(doc_size-1)	# dans row on ajoute le numero de la ligne
		column.append(int(feat[0]))		# on ajoute dans column le premier element de feat
		freq.append(int(feat[1]))		# on ajoute dans freq le deuxieme element de feat

vocab_size=np.max(column)	# vocab_size est le maximum de column
max_class=np.max(row)		# max_class est le max de row
print(vocab_size)		# on affiche vocab_size

# all_lab.append(1)

# on stocke dans une matrice particuliere: (numero du document, numero du terme) frequence d'apparition du terme dans ce document
reuters=sparse.csr_matrix((freq,(row,column)),shape=(doc_size,vocab_size+1))
#print(reuters)


# -------------------------------- Overall Aim --------------------------------

# Question 1)
# The dimension of the problem is 141144. 

# initialize the tab of categories: size 29
doc_per_class = [0 for _ in range(0, 29)]

# fill the tab to get the number of documents for each class
for element in all_lab:
	doc_per_class[element-1] += 1

# Question 2)
print(doc_per_class)

# The number of documents per class is in the following list:
# [5894, 1003, 2472, 2207, 6010, 2992, 1586, 1226, 2007, 3982, 7757, 3644, 3405, 2307, 1040, 1460, 1191, 1733, 4745, 1411, 1016, 3018, 1050, 1184, 1624, 1296, 1018, 1049, 1376]

# Question 3) 
# Split randomly the collection into a training set (52500 documents) and a test set (18203 documents)
training_set, test_set, training_lab, test_lab = train_test_split(reuters, all_lab, train_size = 52500 , test_size = 18203)


# ----------------- Bernoulli vs Multonomial distributions ------------------

# Question 1)

# Purpose of Laplace smoothing: when we use counts to estimate parameters, which can lead to zero values, we should use smoothing. It is equivalent to imposing a uniform prior over our events.

# For example, if we have one document which was strongly classified in one class, and an other which is classified differently because of a word having a probability of zero.

# Therefore, our estimate will tell us that the probability is equal to zero. Now, clearly, this is not true. The probability of this event is low, but it is not zero. Further, because we are multiplying all the probabilities, even one such zero probability term will lead to the entire process failing.

# As a conclusion, we need smoothing — the goal is to increase the zero probability values to a small positive number.


# The Bernoulli Model
# Laplace smoothing: alpha=1.0
bernoulli_model = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

# Training of the Bernoulli Model
bernoulli_model.fit(training_set, training_lab)


# Question 2)
# The Multinomial Model
multinomial_model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# Training of the Multinomial model
multinomial_model.fit(training_set, training_lab)


#  3) Prediction with both models

predict_bernoulli = bernoulli_model.predict(test_set)
predict_multinomial = multinomial_model.predict(test_set)


# Estimation of accuracy of both models on the test set
score_bernoulli = bernoulli_model.score(test_set, test_lab)
print(score_bernoulli)      # 0.5512278195901774

score_multinomial = multinomial_model.score(test_set, test_lab)
print(score_multinomial)    # 0.7721804098225568


#  4) 5-folds Cross-validation 

# Cross-Validation is a technique used in model selection to better estimate the test error of a predictive model. The principle of K-fold cross validation method is to divide randomly the data into K subsets where K-1 subsets are used as the training sets and the remaining subset is used as a validation subset to compute a prediction error. 

# We repeat this procedure K times by considering each subset as the validation subset and the other subsets as the training sets. The procedure is finished when each subset has been used once as the validation subset. The K results from the folds can then be averaged to produce a single estimation.

# The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.


kf=KFold(n_splits=5)

bernoulliModel = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
multinomialModel = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

score_method = "accuracy"

results_bernoulli = cross_val_score(bernoulliModel, reuters, all_lab, cv=kf, scoring=score_method)
results_multinomial = cross_val_score(multinomialModel, reuters, all_lab, cv=kf, scoring=score_method)

#print(results_bernoulli)
#print(results_multinomial)

print(results_bernoulli.mean())     # 0.5580809851318611
print(results_multinomial.mean())   # 0.7691610459499283


# CONCLUSION: 

# As we can see, in this case, the Multinomial model is more efficient than the bernoulli model. Indeed, the bernoulli model is more appropriate when all our features are binary such that they take only two values. Means 0s can represent “word does not occur in the document” and 1s as "word occurs in the document". 
# Whereas, the multinomial model is used when we have discrete data like here we have the count of each word to predict the class or label. 
# So, the results show well that the multinomial model is more appropriate for us.







