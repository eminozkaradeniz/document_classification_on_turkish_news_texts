import time
import pandas as pd
import os
import re
import pickle
from string import digits
from zemberek import TurkishMorphology
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

start_program = time.time()

# Reading STOPWORDS
with open('../stop-words/turkish.txt', mode='r', encoding='utf-8') as f:
    STOPWORDS = f.read().splitlines()

# Reading the dataset
labels = os.listdir('../dataset/train')
entries = []

for label in labels:
    docs = os.listdir(f'../dataset/train/{label}')
    for doc in docs:
        with open(f'../dataset/train/{label}/{doc}', mode='r', encoding='windows-1254') as f:
            entries.append([doc, label, f.read()])

df_train = pd.DataFrame(data=entries, columns=['doc_name', 'label', 'doc'])
df_train.set_index(['doc_name'], inplace=True)

labels = os.listdir('../dataset/test')
entries = []

for label in labels:
    docs = os.listdir(f'../dataset/test/{label}')
    for doc in docs:
        with open(f'../dataset/test/{label}/{doc}', mode='r', encoding='windows-1254') as f:
            entries.append([doc, label, f.read()])

df_test = pd.DataFrame(data=entries, columns=['doc_name', 'label', 'doc'])
df_test.set_index(['doc_name'], inplace=True)

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)

# Creating a TurkishMorphology object from Zemberek library
tm = TurkishMorphology.create_with_defaults()

# Function to remove digits from a text
remove_digits = str.maketrans('', '', digits)


def get_root(word: str):
    try:
        return tm.analyze(word).analysis_results[0].item.root
    except IndexError:
        return word


def text_process(text: str) -> str:
    text = text.lower()
    text = text.translate(remove_digits)
    text = re.sub(r'[^\w\s]', '', text)
    words = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return ' '.join([get_root(word) for word in words])


# Processing every news text in the dataset
# lowercase the text -> removing digits, punctuations and stopwords ->
# changing words to their roots (Stemming)
start = time.time()
df_train['doc'] = df_train['doc'].apply(text_process)
df_test['doc'] = df_test['doc'].apply(text_process)
print('text_process: {} sn'.format(time.time() - start))

# Train and Test Set
X_train, X_test, y_train, y_test = df_train['doc'], df_test['doc'], df_train['label'], df_test['label']

# Creating Sparse Matrix
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Creating and running the model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Printing the performance measurement values of the model
print("\nMultinomial Naive Bayes:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Saving Model
pickle.dump(model, open("model.sav", 'wb'))
pickle.dump(vectorizer, open("vectorizer.pk", 'wb'))

print('time: ', time.time() - start_program)
