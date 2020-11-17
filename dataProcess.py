import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

train_list = []
train_label = []
for label in os.listdir("C50/C50train"):
    for file in os.listdir("C50/C50train/" + label):
        with open(os.path.join("C50/C50train/",label, file), "r") as f:
            text = f.read()
            train_list.append(text)
            train_label.append(label)

test_list = []
test_label = []
for label in os.listdir("C50/C50test"):
    for file in os.listdir("C50/C50test/" + label):
        with open(os.path.join("C50/C50test/",label, file), "r") as f:
            text = f.read()
            test_list.append(text)
            test_label.append(label)

docs = train_list + test_list
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df.drop([col for col, val in df.sum().iteritems() if val < 50], axis=1, inplace=True)
