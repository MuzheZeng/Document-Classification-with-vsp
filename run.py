import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

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

df.drop([col for col, val in df[:2500,:].sum().iteritems() if val < 50], axis=1, inplace=True)

df['label'] = train_label + test_label
df['label'] = df['label'].astype('category')
df.drop(columns=[w for w in df.columns if w in stopwords.words('english')], inplace=True)




from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R), R

X_train, y_train, X_test, y_test = df.iloc[:2500,:-1], df.iloc[:2500, -1], df.iloc[2500:, :-1], df.iloc[2500:, -1]

sts = StandardScaler()
X = sts.fit_transform(X_train)
y = y_train.values

svm = SVC(gamma="auto")
logi = LogisticRegression(solver="lbfgs", multi_class="multinomial",C=1)
mlp = MLPClassifier(batch_size = 100)
models = [svm, logi, mlp]

for clf in models:
    scores = cross_val_score(clf, X, y, scoring="accuracy", cv=5)
    print(clf.__class__.__name__, np.mean(scores), np.std(scores))


# Grid Search CV
from sklearn.model_selection import GridSearchCV

param_grid = [{
    'C':[0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'sigmoid'],
}]

svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", return_train_score=True)
grid_search.fit(X, y)

print(grid_search.best_params_)


svc_best = SVC(C=10, kernel='sigmoid')
svc_best.fit(X_train, y)
final_pred = svc_best.predict(sts.transform(X_test))
accuracy_score(final_pred, y_test.values)


param_grid2 = [{
    'hidden_layer_sizes':[(100,),(200,),(100,100), (200,200)],
    'activation': ['logistic', 'relu'],
    'alpha':[0.01, 0.1, 1],
}]

mlp
grid_search2 = GridSearchCV(mlp, param_grid2, cv=5, scoring="accuracy", return_train_score=True)
grid_search2.fit(X, y)


mlp_best = MLPClassifier(hidden_layer_sizes=(100,100), activation='logistic', alpha=0.1)
mlp_best.fit(X_train, y)
final_pred2 = mlp_best.predict(sts.transform(X_test))
accuracy_score(final_pred2, y_test.values)


logi = LogisticRegression()
logi.fit(X_train, y)
final_pred3 = logi.predict(sts.transform(X_test))
accuracy_score(final_pred3, y_test.values)



# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(18,15))
sns.heatmap(confusion_matrix(final_pred2, y_test.values)).set_title("Confusion Matrix for the ANN Prediction", fontsize=30)




# vsp, abandoned for lack of space in the report
# k = 200
# X_svd = TruncatedSVD(n_components=k)
# X_svd.fit(X_train.values)
# plt.plot(range(len(X_svd.singular_values_)), X_svd.singular_values_)

# factors = X_train.values @ X_svd.components_.T
# vari = varimax(factors, q=k)

# for clf in models:
#     scores = cross_val_score(clf, vari[0], y, scoring="accuracy", cv=5)
#     print(clf.__class__.__name__, np.mean(scores), np.std(scores))
