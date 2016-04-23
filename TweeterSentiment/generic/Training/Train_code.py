import numpy
import re
import random
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier

negation_matrix = ["don't","never", "nothing", "nowhere", "noone", "none", "not","hasn't","hadn't","can't","couldn't","shouldn't","won't","wouldn't","don't","doesn't","didn't","isn't","aren't","ain't","cannot","shant","cant","dont"];



# def tag_word(tweet):
# 	word_list = re.split(r'[:,?; ]',tweet)
# 	resulte = [];
# 	for i in word_list:
# 		if i in negation_matrix:
# 			resulte.append(i)
# 	for result in resulte:		
# 		if(result != ''):
# 			partial = re.sub(r'( [A-Za-z]+)',r'NEG_\1',tweet[tweet.find(result)+len(result):],2)
# 			tweet =  tweet[:tweet.find(result)+len(result)]+partial
# 		else:
# 			return tweet
# 	return tweet


def tag_word(tweet):
	word_list = re.split(r'[:,?; ]',tweet)
	result = '';
	for i in word_list:
		if i in negation_matrix:
			result = i
		
	if(result != ''):
		partial = re.sub(r'( [A-Za-z]+)',r' NEG_\1',tweet[tweet.find(result)+len(result):],2)
		return tweet[:tweet.find(result)+len(result)]+partial
	else:
		return tweet


def preprocess_tweet(tweet):
	tweet = re.sub('@[\w]*','',tweet);
	tweet = re.sub(r'(.)\1{2,}',r'\1\1\1',tweet)
	tweet = re.sub(r'[ ]+',' ',tweet);
	tweet = re.sub('https?://[^\s]*','',tweet);
	tweet = re.sub('www.[^\s]*','',tweet);
	tweet = re.sub('^ +','',tweet);

	 #print tweet
	tweet = tag_word(tweet)
	tweet = re.sub(r'\'ve',' have',tweet)
	tweet = re.sub(r'I\'m','I am',tweet)
	tweet = re.sub(r'\'s',' is' ,tweet)
	tweet  = re.sub(r'\'re',' are',tweet)
	tweet = re.sub(r'n\'t',' not',tweet)
	 #print tweet
	# tweet.decode("utf8","replace")
	return tweet;

def read_data(training_tweets,training_label,validation_label,validation_tweets):
	f1 = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/training_data_gen.csv','r')
	f2 = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/validation_data_gen.csv','r')
	f3 = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/testing_data_gen.csv','r')
	for line in f1:
 		lines = line.split(',',1)
 		training_tweets.append(preprocess_tweet(lines[1]))
 		training_label.append(lines[0][1:-1])
 	for line in f2:
 		lines = line.split(',',1)
 		training_tweets.append(preprocess_tweet(lines[1]))
 		training_label.append(lines[0][1:-1])


 	for line in f3:
 		lines = line.split(',',1)
 		training_tweets.append(preprocess_tweet(lines[1]))
 		training_label.append(lines[0][1:-1])
 	f1.close()
 	f2.close()
 	f3.close()
 	f4 = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/preprocessed_training_tweet.csv','w')
 	for line in training_tweets:
 		f4.write(line)
 		#f4.write('\n')
 	f4.close()


  	f1 = open('/Users/kushaagragoyal/Downloads/Sentiment Analysis Dataset.csv','r')
	f1.readline()
	counter1 = 0
	counter2 = 0
	for line in f1:
  		lines = line.split(',',3)
  		validation_tweets.append(preprocess_tweet(lines[3]))
  		if(lines[1] == '0'):
  			validation_label.append(lines[1])
  			counter1 = counter1+1
  		else:
  			validation_label.append('4')
  			counter2 = counter2+1
	print counter1
	print counter2
	f4 = open('/Users/kushaagragoyal/Desktop/COL772/Assignment 1/preprocessed_testing_tweet.csv','w')
 	for line in training_tweets:
 		f4.write(line)
 		f4.write('\n')
 	f4.close()


def token_count_feature_unigram(training_tweets,training_label):
	vectorizer = CountVectorizer(min_df = 5, encoding = 'latin_1',binary = True,ngram_range = (1,2))
	X = vectorizer.fit_transform(training_tweets)
	return X,vectorizer;


def token_count_feature_bigram(training_tweets,training_label):
	vectorizer = CountVectorizer(min_df = 1, encoding = 'latin_1',ngram_range = (1,2))
	X = vectorizer.fit_transform(training_tweets)
	return X,vectorizer;

def token_presence_feature_unigram(training_tweets,training_label):
	vectorizer = HashingVectorizer(encoding = 'latin_1')
	X = vectorizer.fit_transform(training_tweets)
	return X,vectorizer;
def token_presence_feature_bigram(training_tweets,training_label):
	vectorizer = HashingVectorizer(encoding = 'latin_1')
	X = vectorizer.fit_transform(training_tweets)
	return X,vectorizer;

def tfidf_feature(training_tweets,training_label):
	vectorizer = TfidfVectorizer(encoding = 'latin_1',ngram_range=(1,2),min_df = 5,max_df = 0.9)
	X= vectorizer.fit_transform(training_tweets)
	return X,vectorizer


def learn_logistic_regression(X,Y):
	classifier = LogisticRegression()
	classifier.fit(X,Y);
	return classifier;

def test_classifier(classifier,validation_tweets,validation_label,vectorizer):
	print classifier.score(vectorizer.transform(validation_tweets),validation_label)
	prediction = classifier.predict(vectorizer.transform(validation_tweets))
	i=0;
	#for value in prediction:
	#	if(value != validation_label[i]):
	#		print validation_tweets[i],prediction[i],validation_label[i]
	#	i = i+1
	print precision_recall_fscore_support(validation_label,prediction)

training_tweets = []
training_label = []
validation_tweets = []
validation_label = []
read_data(training_tweets,training_label,validation_label,validation_tweets)
print 'First Gentleman'
#X,vectorizer = token_count_feature_unigram(training_tweets,training_label)

#training_tweets = training_tweets + validation_tweets
#training_label = training_label + validation_label
#training_tweets = training_tweets[:1500000]
#training_label = training_label[:1500000]
#validation_tweets = training_tweets[1500000:]
#validation_label = training_tweets[1500000:]

X,vectorizer = tfidf_feature(training_tweets,training_label)
print 'Gentleman'

#classifier = BaggingClassifier(svm.LinearSVC(),max_samples = 0.5,n_jobs = -1)
#classifier.fit(X,training_label);
#scores = cross_validation.cross_val_score(classifier,X,training_label,cv=10)


#classifier3 = svm.LinearSVC();
#classifier3.fit(X,training_label);
#scores = cross_validation.cross_val_score(classifier,X,training_label,cv=10)

#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print 'yo'
classifier2 = LogisticRegression();
classifier2.fit(X,training_label);
#scores = cross_validation.cross_val_score(classifier2,X,training_label,cv=10)

#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#print 'Yo'
#classifier2 = learn_logistic_regression(X,training_label)
#print 'ready guys'
#test_classifier(classifier,validation_tweets,validation_label,vectorizer)
#test_classifier(classifier,training_tweets,training_label,vectorizer)

#test_classifier(classifier3,validation_tweets,validation_label,vectorizer)
#test_classifier(classifier3,training_tweets,training_label,vectorizer)


#test_classifier(classifier2,validation_tweets,validation_label,vectorizer)
test_classifier(classifier2,training_tweets,training_label,vectorizer)

#classifier3 = RandomForestClassifier(n_estimators = 3)
#classifier3.fit(X,training_label)
#test_classifier(classifier3,validation_tweets,validation_label,vectorizer)
#test_classifier(classifier3,training_tweets,training_label,vectorizer)

joblib.dump(classifier2,'classifier_Logistic.pkl')
#joblib.dump(classifier,'classifier_svm.pkl')
#joblib.dump(classifier3,'classifier_ensemble.pkl')
joblib.dump(vectorizer,'vector.pkl')


#print classifier.predict(vectorizer.transform([preprocess_tweet('You are a good boy'),preprocess_tweet('You are not a good boy'),preprocess_tweet('You foolosh Gentelemt'),preprocess_tweet('Asshole'),preprocess_tweet('randomness gadha stud bhai'),preprocess_tweet('I shall rule    thy worldi @obama')]))

