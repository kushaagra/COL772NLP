import numpy
import re
import random
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support

# a = numpy.zeros(5)
# print a
f = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/training.csv','r')
input_training_data = []
training_tweets=[]
training_label = []

validation_tweets = []
validation_label = []

testing_tweets = []
testing_label = []

for line in f:
    input_training_data.append(line);
   
print len(input_training_data)
random.shuffle(input_training_data)
random.shuffle(input_training_data)

training_data = input_training_data[0:1000000]
# testing_data = input_training_data[1000000:1599999]
number_of_positive_labels = 0;
number_of_negative_labels = 0;
for line in training_data:
 	lines = line.split(',')
 	training_tweets.append(lines[1][1:-2])
 	training_label.append(lines[0][1:-1])
 	#print training_label[len(training_label)-1]
 	if (training_label[len(training_label)-1] == '4'):
 		number_of_positive_labels  = number_of_positive_labels + 1
 	if(training_label[len(training_label)-1] == '0'):
 		number_of_negative_labels = number_of_negative_labels+1;
print number_of_positive_labels
print number_of_negative_labels

f = open('training_data_gen.csv','w')
for line in training_data:
	f.write(line)
f.close()

validation_data = input_training_data[1000000:1300000]


number_of_positive_labels = 0;
number_of_negative_labels = 0;
for line in validation_data:
 	lines = line.split(',')
 	validation_tweets.append(lines[1][1:-2])
 	validation_label.append(lines[0][1:-1])
 	#print training_label[len(training_label)-1]
 	if (validation_label[len(validation_label)-1] == '4'):
 		number_of_positive_labels  = number_of_positive_labels + 1
 	if(validation_label[len(validation_label)-1] == '0'):
 		number_of_negative_labels = number_of_negative_labels+1;
print number_of_positive_labels
print number_of_negative_labels

f = open('validation_data_gen.csv','w')
for line in validation_data:
	f.write(line)
f.close()

testing_data = input_training_data[1300000:1600000]


number_of_positive_labels = 0;
number_of_negative_labels = 0;
for line in testing_data:
 	lines = line.split(',')
 	testing_tweets.append(lines[1][1:-2])
 	testing_label.append(lines[0][1:-1])
 	#print training_label[len(training_label)-1]
 	if (testing_label[len(testing_label)-1] == '4'):
 		number_of_positive_labels  = number_of_positive_labels + 1
 	if(testing_label[len(testing_label)-1] == '0'):
 		number_of_negative_labels = number_of_negative_labels+1;
print number_of_positive_labels
print number_of_negative_labels

f = open('testing_data_gen.csv','w')
for line in testing_data:
	f.write(line)
f.close()

# for line in testing_data:
# 	lines = line.split(',')
# 	testing_tweets.append(lines[1][1:-2])
# 	testing_label.append(lines[0][1:-1])

# tokeniser = TweetTokenizer(preserve_case = False)
# #for i in range(10):
#  #   print training_tweets[i]
#     #print training_tweets[i].split(' ')
#   #  print tokeniser.tokenize(training_tweets[i])
# #print tweets

# print training_tweets[1:10]

# vectorizer = CountVectorizer(min_df = 1, encoding = 'latin_1',tokenizer = tokeniser.tokenize)
# X = vectorizer.fit_transform(training_tweets)
# Y = training_label
# classifier = MultinomialNB()
# classifier.fit(X,Y)
# # #classifier.predict(X[30])
# A = classifier.score(vectorizer.transform(testing_tweets),testing_label)
# B = classifier.score(vectorizer.transform(training_tweets),training_label)


# classifier2 = LogisticRegression()
# classifier2.fit(X,Y)
# A1 = classifier2.score(vectorizer.transform(testing_tweets),testing_label)
# B1 = classifier2.score(vectorizer.transform(training_tweets),training_label)



# vectorizer = CountVectorizer(min_df = 5, encoding = 'latin_1',stop_words = 'english')
# X = vectorizer.fit_transform(training_tweets)
# Y = training_label
# classifier = MultinomialNB()
# classifier.fit(X,Y)
# # #classifier.predict(X[30])
# A3 = classifier.score(vectorizer.transform(testing_tweets),testing_label)
# B3 = classifier.score(vectorizer.transform(training_tweets),training_label)


# classifier2 = LogisticRegression()
# classifier2.fit(X,Y)
# A4 = classifier2.score(vectorizer.transform(testing_tweets),testing_label)
# B4 = classifier2.score(vectorizer.transform(training_tweets),training_label)



#joblib.dump(classifier2,'log.pkl')
#joblib.dump(vectorizer,'vector.pkl')
#classi = joblib.load('log.pkl')
#vecto = joblib.load('vector.pkl')
#A5 = classi.score(vecto.transform(testing_tweets),testing_label)
#Y_Pred = classi.predict(vecto.transform(testing_tweets))
#precision_recall_fscore_support(testing_label,Y_Pred)




 