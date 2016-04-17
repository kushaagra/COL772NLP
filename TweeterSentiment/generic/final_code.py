import sys
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

input_file = sys.argv[1]
output_file = sys.argv[2]

negation_matrix = ["don't","never", "nothing", "nowhere", "noone", "none", "not","hasn't","hadn't","can't","couldn't","shouldn't","won't","wouldn't","don't","doesn't","didn't","isn't","aren't","ain't","cannot","shant","cant","dont"];

def tag_word(tweet):
	word_list = re.split(r'[:,?; ]',tweet)
	result = '';
	for i in word_list:
		if i in negation_matrix:
			result = i
			break
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
	 #print tweet
	# tweet.decode("utf8","replace")
	 return tweet;



def read_data(testing_tweets):
	f1 = open(input_file,'r')
	for line in f1:
 		testing_tweets.append(preprocess_tweet(line))

 	f1.close()
testing_tweets = []
read_data(testing_tweets)
classifier = joblib.load('classifier_Logistic.pkl')
vectorizer = joblib.load('vector.pkl')
print 'Classifier loaded'
prediction = classifier.predict(vectorizer.transform(testing_tweets))
f = open(output_file,'w')
for i in prediction:
	f.write(i)
	f.write("\n")
f.close()


