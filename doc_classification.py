import os
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

doc_directory = r'C:\Users\creep\BizAmica\document_classifcation\docs_by_type'

file_list = []

text_and_labels = []

text_list = []

for root, _, files in os.walk(doc_directory):
    for file in files:
        file_list.append(os.path.join(root, file))
        with open(os.path.join(root, file), 'r') as f:
            if 'non-airbus' in root:
                text_and_labels.append([file, 'non-airbus'])
                text_list.append(f.read())
            else:
                text_and_labels.append([file, 'airbus'])
                text_list.append(f.read())
random.shuffle(text_and_labels)

train_df = pd.DataFrame(text_and_labels, columns=['Text','Label'])

print(train_df)

plt.subplots(figsize=(15,5))
sns.countplot(data=train_df, x='Label', order=train_df['Label'].value_counts().index)

# plt.show()

print(text_list)

df = pd.DataFrame(text_list, columns=['Text'])

print(df.shape)

df = df.replace('\n', '', regex = True)
print(df.head(10))

print(df.isnull().sum(axis = 1))

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

bow_transformer = CountVectorizer().fit(df['Text'])
messages_bow = bow_transformer.transform(df['Text'])

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)
print("messages_tfidf shape: ",messages_tfidf.shape)

from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
train_df.iloc[:,1]=label_x.fit_transform(train_df.iloc[:,1])

# y = train_df['Label']
y = pd.to_numeric(train_df['Label'])

# print(y)

print("value counts ",y.value_counts())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(messages_tfidf,y,test_size=0.2,random_state=101, stratify = y)

print(y)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

params = {'bootstrap': [True, False],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['log2', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

rfc=RandomForestClassifier(n_estimators = 350, random_state = 0)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(rfc, param_distributions=params, scoring = 'accuracy', n_iter=param_comb, n_jobs=4, cv=skf.split(messages_tfidf,y), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(messages_tfidf, y)
timer(start_time) # timing ends here for "start_time" variable

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('rfc-random-grid-search-results-01.csv', index=False)

rfc = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=60, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=1200,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)

rfc.fit(X_train, y_train)


y_pred_rfc = rfc.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))

from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
train_df.iloc[:,1]=label_x.fit_transform(train_df.iloc[:,1])
y = pd.to_numeric(train_df['Label'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['Text'],y,test_size=0.2,random_state=101, stratify = y)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', rfc)  # train on TF-IDF vectors w/ Random Forest Classifier
     ])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

orig = label_x.inverse_transform(y)

print(orig)

from joblib import dump
dump(pipeline, 'Doc_Classify_Model_rfc.pkl')

['Doc_Classify_Model_rfc.pkl']

