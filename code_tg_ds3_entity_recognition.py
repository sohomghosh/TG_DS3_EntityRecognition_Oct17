import pandas as pd
import numpy as np
#from gensim.models import word2vec
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter
import xgboost as xgb
from sklearn.metrics import f1_score


train=pd.read_csv("/home/sohom/Desktop/TechGig_DS3/train.csv")
test=pd.read_csv("/home/sohom/Desktop/TechGig_DS3/test.csv")

train_test=train.append(test)
sentences_split=[re.split('\.| |\,|\:|\\r', i) for i in train_test['description']]
classes=sorted(set(train['StringToExtract']))

classes_dict = {}
classes_dict_rev = {}
for j in range(0,len(classes)):
	classes_dict[classes[j]]=j
	classes_dict_rev[j]=classes[j]

train['StringToExtract_encoded'] = train['StringToExtract'].replace(to_replace=classes_dict)
def my_tokenizer(s):
	return re.split('\.| |\,|\:|\\r', s)


####https://youtu.be/al82wLfSRoA
####Disjoint CRF, Joint CRF, Cascaded CRF ::: combine char level emebdding beside w2v -->> pass through LSTM -->> Take softmax/ CRF  -->> Label as 'StringToExtract' -->> Error measure by f1 score 


sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.001, use_idf=True, smooth_idf=False, tokenizer=my_tokenizer,sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit(train_test['description'])
train_test_tfidf=pd.DataFrame(sklearn_tfidf.transform(train_test['description']).todense())
#sklearn_representation.vocabulary_

X_train_all=train_test_tfidf[0:len(train.index)]
X_train_all['label']=train['StringToExtract_encoded']

X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)
X_test=train_test_tfidf[len(train.index):len(train_test.index)]

features=X_test.columns

dtrain = xgb.DMatrix(X_train[features], X_train['label'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features], X_valid['label'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)


nrounds = 200
watchlist = [(dtrain, 'train')]
#########num_classes change
params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4,"num_class": len(set(X_train_all['label'])), "silent": 1,"eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)

print(f1_score(valid_preds,X_valid['label'],average='weighted'))

test_preds_final=[classes_dict_rev[i] for i in test_preds]

submit = pd.DataFrame({'id': test['id'], 'StringToExtract': test_preds_final})
submit[['id','StringToExtract']].to_csv("XGB2.csv", index=False)


############################################################################################################################
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

create_feature_map(features)
bst.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
imp_df.to_csv("imp_feat.txt", index=False)


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')


#imp_df = pd.read_csv('imp_feat.txt')
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score")
ax.set_title("Variable importance")
autolabel(rects)
plt.show()
############################################################################################################################


####################################################################################################################################
##########################################OTHER APPROACHES##########################################################################
####################################################################################################################################

#3) Use word2vec with proper tokenization. For each word find the position in vector space. For each word of each sentence in the test set find its distance from labels mentioned in the training set list (or centroid of all labels mentioned in the training set). The word nearest to this centroid is the word to be detected.
#4) Other Features: Position of the word from starting, end of a sentence
#5) If any of sentence is same as that of the train_set then use it as it is
#6) Make a list of words before and after the "StringToExtract" within a fixed window size
#7) Number of times any of the "StringToExtract" of training set is occuring in the sentence of the test set
#8) use fasttext classifier; tags "StringToExtract" preceed by __label__


model = fasttext.cbow('data.txt', 'model')
classifier = fasttext.supervised('data.train.txt', 'model', label_prefix='__label__')
labels = classifier.predict(texts, k)



'''
####################################################################################################################################
##########################################EXPERIMENTATION###########################################################################
####################################################################################################################################

sentences_split_test=[re.split('\.| |\,|\:|\\r', i) for i in test['description']]
sentences_split_train=[re.split('\.| |\,|\:|\\r', i) for i in train['description']]

tags=['dir', 'kno', 'onl', 'wcm', 'pdu', '10.', 'cn-', 'sfr', 'pbu', 'cas', 'hdr', 'phk', 'mil', 'bei', 'pbr', 'muc', 'us-', 'ccs', 'lux', 'lch', 'pro', 'pca', 'err', 'lpp', 'con', 'ppr', 'lpa', 'plu', 'esa', 'tim', 'dis', 'asi', 'dc=', 'par', 'buc', 'lpg', 'dea', 'pps', 'syd', 'pvs', 'ams', 'bkk', 'bar', 'pny', 'ser', 'sgd', 'sin', 'aud', 'au-', 'dus', 'lub', 'pra', 'doc', 'del', 'jp-', 'seo', 'tok', 'ldt', 'atl', 'spr', 'ame', 'lpm', 'psh', 'pse', 'ce_', 'esd', 'rom', 'psy', 'ind', 'int', 'doh', 'lpd', 'cck', 'eur', 'esc', 'com', 'ddl', 'hyp', 'ist', 'blr', 'web', 'fra', 'mad', 'pty', '6.1', 'nje', 'lsl', 'aut', 'wil', 'eve', 'ppa', 'pbl', 'ban', 'riy', 'war', 'alu', 'sap', 'pbc', 'was', 'gpm', 'lpl', 'nyc', 'pfr', 'per', 'hkg', 'mos', 'dub', 'ldv', 'shn', 'epi', 'pbk', 'lpi', 'pmu', 'bru', 'lst', 'grd', 'pdx']

[set(i).intersection(set(classes)) for i in sentences_split][:7]
#[{'ddlsql144', 'internal'}, {'author', 'ddlsql43', 'internal'}, {'author', 'ddlsql43', 'internal'}, {'internal', 'knowwho', 'ddlsql43'}, {'fra-sql-03', 'internal'}, {'au-per-06a-stwp-01'}, {'ddl-mb-08', 'internal'}]

list(train_test['StringToExtract'])[:7]
#['ddlsql144', 'ddlsql43', 'ddlsql43', 'ddlsql43', 'fra-sql-03', 'au-per-06a-stwp-01', 'ddl-mb-08']


for i in sentences_split[:7]:
	k=''
	for j in i:
		for tag in tags:
			if str(j)[:3].lower()==tag:
				k=k+' '+j
	print(k+"\n")


Counter([len(set(i).intersection(set(classes)))==0 for i in sentences_split])
#Counter({False: 30864, True: 56})
[i for i in sentences_split if len(set(i).intersection(set(classes)))==0]

[[sentences_split_train[i],list(train['StringToExtract'])[i],list(train['description'])[i]] for i in range(0,len(sentences_split_train)) if len(set(sentences_split_train[i]).intersection(set(classes)))==0]
#\ Missing in the regular expression for splitting; 'grdp' ; name: grdp\r\r\nseverity: 

Counter([list(train['StringToExtract'])[i] for i in range(0,len(sentences_split_train)) if len(set(sentences_split_train[i]).intersection(set(classes)))==0])

'''


#####################################################################################################################################
###############################################SPACY NER EXPERIMENTATION#########################################################
#####################################################################################################################################
'''
#Source: https://spacy.io/docs/usage/entity-recognition

import spacy
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer

train_data = [
    ('Who is Chaka Khan?', [(7, 17, 'PERSON')]),
    ('I like London and Berlin.', [(7, 13, 'LOC'), (18, 24, 'LOC')])
]

nlp = spacy.load('en', entity=False, parser=False)
ner = EntityRecognizer(nlp.vocab, entity_types=['PERSON', 'LOC'])

for itn in range(5):
    random.shuffle(train_data)
    for raw_text, entity_offsets in train_data:
        doc = nlp.make_doc(raw_text)
        gold = GoldParse(doc, entities=entity_offsets)

        nlp.tagger(doc)
        ner.update(doc, gold)
ner.model.end_training()
'''

#####################################################################################################################################
###############################################DEEP LEARNING EXPERIMENTATION#########################################################
#####################################################################################################################################
'''
#Source: https://github.com/SujathaSubramanian/Projects/blob/master/SentimentalAnalysis/USAirlineSentimentAnalysis/TwitterUSAirlineSentimentAnalysis.ipynb

import sklearn
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
import re
import nltk
import tensorflow as tf

lstm_size = 256
lstm_layers = 1
batch_size = 100
learning_rate = 0.001

#Create input placeholders
n_words = len(vocab_to_int)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
# Embedding - Efficient way to process the input vector is to do embedding instead of one-hot encoding
# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    
#Build the LSTM cells
with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

#RNN Forward pass
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)
    
### Output - Final output of the RNN layer will be used for sentiment prediction. 
### So we need to grab the last output with `outputs[:, -1]`, the calculate the cost from that and `labels_`.
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
## Graph for checking Validation accuracy
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

### Batching - Pick only full batches of data and return based on the batch_size
def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/twitter_sentiment.ckpt")


## Testing
#For the tweets in test set, predict the sentiment using the trained model                 

test_acc = []
test_pred = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state= sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
        prediction = tf.cast(tf.round(predictions),tf.int32)
        prediction = sess.run(prediction,feed_dict=feed)
        test_pred.append(prediction)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
    
    
##Use the tweet sentiment predicted for the data in the test set,for plotting the wordcloud
test_pred_flat = (np.array(test_pred)).flatten()
start_idx = len(train_x) + len(val_x)
end_idx = start_idx + len(test_pred_flat)+1
Tweet.loc[start_idx:end_idx,'predicted_sentiment'] = test_pred_flat
'''
