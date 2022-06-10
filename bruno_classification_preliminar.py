# %%
import pandas as pd
# %%
import numpy as np
# %%
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1)
# %%
from sklearn.model_selection import cross_val_score, cross_val_predict
# %%
from sklearn.linear_model import LogisticRegression
# %%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# %%
mnist.keys()
# %%
X_bruno, y_bruno = mnist.data, mnist.target
# %%
X_bruno.dtypes
# %%
y_bruno.dtypes
# %%
some_digit1, some_digit2, some_digit3 = X_bruno.iloc[7], X_bruno.iloc[5], X_bruno.iloc[0]
# %%
some_digit1
# %%
import matplotlib.pyplot as plt
# %%
some_digit1_image = some_digit1.values.reshape(28, 28)
some_digit2_image = some_digit2.values.reshape(28, 28)
some_digit3_image = some_digit3.values.reshape(28, 28)
# %%
plt.imshow(some_digit1_image, cmap='cividis')
# %%
plt.imshow(some_digit2_image, cmap='cividis')
# %%
plt.imshow(some_digit3_image, cmap='cividis')
# %%
y_bruno = y_bruno.astype(np.uint8)
# %%
y_bruno[:5]
# %%
y_bruno = np.where(y_bruno < 4, 0, np.where(((y_bruno >= 4) & (y_bruno <7)), 1, 9))
# %%
y_bruno = pd.Series(y_bruno)
# %%
y_bruno[:5]
# %%
pd.Series(y_bruno).unique()
# %%
y_bruno.value_counts()
# %%
y_bruno.value_counts().plot(kind='bar')
plt.xticks(rotation=0)
plt.show()
# %%
X_bruno_train, X_bruno_test, y_bruno_train, y_bruno_test = X_bruno[:50000], X_bruno[50000:], y_bruno[:50000], y_bruno[50000:]
# %%
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
# %%
NB_Bernoulli_cl_bruno = BernoulliNB()
# %%
NB_MulinomialNB_cl_bruno = MultinomialNB()
# %%
NB_GaussianNB_cl_bruno = GaussianNB()
# %%
scores_Bernoulli = cross_val_score(NB_Bernoulli_cl_bruno, X_bruno_train, y_bruno_train, cv=3)
# %%
def add_labels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
# %%
x = ['cv_1', 'cv_2', 'cv_3']
plt.bar(x, scores_Bernoulli)
plt.yticks(np.arange(0, 1, step=0.05))
add_labels(x, np.round(scores_Bernoulli, 4))
plt.title('Bernoulli Naive Bayes')
plt.show()
# %%
scores_Multinomial = cross_val_score(NB_MulinomialNB_cl_bruno, X_bruno_train, y_bruno_train, cv=3)
# %%
plt.bar(x, scores_Multinomial)
plt.yticks(np.arange(0, 1, step=0.05))
add_labels(x, np.round(scores_Multinomial, 4))
plt.title('Multinomial Naive Bayes')
plt.show()
# %%
scores_Gaussian = cross_val_score(NB_GaussianNB_cl_bruno, X_bruno_train, y_bruno_train, cv=3)
# %%
plt.bar(x, scores_Gaussian)
plt.yticks(np.arange(0, 1, step=0.05))
add_labels(x, np.round(scores_Gaussian, 4))
plt.title('Gaussian Naive Bayes')
plt.show()
# %%
NB_Bernoulli_cl_bruno.fit(X_bruno_train, y_bruno_train)
# %%
y_train_bernoulli_pred = cross_val_predict(NB_Bernoulli_cl_bruno, X_bruno_train, y_bruno_train, cv=3)
# %%
cf = confusion_matrix(y_bruno_train, y_train_bernoulli_pred)
print(cf)
# %%
print(classification_report(y_bruno_train, y_train_bernoulli_pred))
# %%
y_train_gaussian_pred = cross_val_predict(NB_MulinomialNB_cl_bruno, X_bruno_train, y_bruno_train, cv=3)

# %%
cf_multinomial = confusion_matrix(y_bruno_train, y_train_bernoulli_pred)
print(cf_multinomial)
# %%
print(classification_report(y_bruno_train, y_train_gaussian_pred))
# %%
print(some_digit1, some_digit2, some_digit3)
# %%
y_pred = NB_Bernoulli_cl_bruno.predict([some_digit1, some_digit2, some_digit3])
# %%
pred_real = pd.DataFrame({'y_real': y_bruno[[7, 5, 0]].to_numpy(), 'y_pred': y_pred}, index = ['some_digit1', 'some_digit2', 'some_digit3'])
# print(y_bruno.iloc[[7, 5, 0]])
# %%
print(pred_real)
# %%
print(y_bruno.iloc[[7, 5, 0]])
# %%
type(y_bruno.iloc[[7, 5, 0]])
# %%
type(y_pred)
# %%
LR_clf_bruno = LogisticRegression(solver='lbfgs', tol = 0.1, max_iter=1200, multi_class = 'multinomial')
# %%
LR_clf_bruno = LogisticRegression(solver='saga', tol = 0.1, max_iter=1200, multi_class = 'multinomial')
# %%
LR_clf_bruno.fit(X_bruno_train, y_bruno_train)

# %%
LR_scores = cross_val_score(LR_clf_bruno, X_bruno_train, y_bruno_train, cv=3)
# %%
plt.bar(x, LR_scores)
plt.yticks(np.arange(0, 1, step=0.05))
add_labels(x, np.round(LR_scores, 4))
plt.title("Logistic Regression 'solver = saga'")
plt.show()
# %%
LR_pred = LR_clf_bruno.predict(X_bruno_test)
# %%
LR_pred_real = pd.DataFrame({'Predicted': LR_pred, 'Test': y_bruno_test.to_numpy(), 'Predicted == Test': LR_pred == y_bruno_test})
# %%
print(LR_pred_real)
# %%
print(LR_pred)
# %%
LR_clf_bruno.score(X_bruno_test, y_bruno_test)
# %%
LR_cf_bruno = confusion_matrix(y_bruno_test, LR_pred)
# %%
print(LR_cf_bruno)
# %%
print(classification_report(y_bruno_test, LR_pred))
# %%    
LR_pred_some = LR_clf_bruno.predict(pd.DataFrame([some_digit1, some_digit2, some_digit3], columns = X_bruno.columns))
# %%
print(pd.DataFrame([some_digit1, some_digit2, some_digit3], columns = X_bruno.columns))
# %%
print(LR_pred_some)
# %%
