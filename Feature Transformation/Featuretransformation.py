import pandas as pd
import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])

df.head()

df.isnull().sum()

df['Age'].fillna(df['Age'].mean(),inplace=True)

df.head()

X = df.iloc[:,1:3]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(X_train['Age'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(X_train['Age'], dist="norm", plot=plt)
plt.title('Age QQ Plot')

plt.show()

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(X_train['Fare'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(X_train['Fare'], dist="norm", plot=plt)
plt.title('Age QQ Plot')

plt.show()

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train,y_train)
clf2.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)

print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))

trf = FunctionTransformer(func=np.log1p)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed,y_train)
clf2.fit(X_train_transformed,y_train)

y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)

print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))

X_transformed = trf.fit_transform(X)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR",np.mean(cross_val_score(clf,X_transformed,y,scoring='accuracy',cv=10)))
print("DT",np.mean(cross_val_score(clf2,X_transformed,y,scoring='accuracy',cv=10)))

plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(X_train['Fare'], dist="norm", plot=plt)
plt.title('Fare Before Log')

plt.subplot(122)
stats.probplot(X_train_transformed['Fare'], dist="norm", plot=plt)
plt.title('Fare After Log')

plt.show()

plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(X_train['Age'], dist="norm", plot=plt)
plt.title('Age Before Log')

plt.subplot(122)
stats.probplot(X_train_transformed['Age'], dist="norm", plot=plt)
plt.title('Age After Log')

plt.show()

trf2 = ColumnTransformer([('log',FunctionTransformer(np.log1p),['Fare'])],remainder='passthrough')

X_train_transformed2 = trf2.fit_transform(X_train)
X_test_transformed2 = trf2.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed2,y_train)
clf2.fit(X_train_transformed2,y_train)

y_pred = clf.predict(X_test_transformed2)
y_pred2 = clf2.predict(X_test_transformed2)

print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred2))

X_transformed2 = trf2.fit_transform(X)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR",np.mean(cross_val_score(clf,X_transformed2,y,scoring='accuracy',cv=10)))
print("DT",np.mean(cross_val_score(clf2,X_transformed2,y,scoring='accuracy',cv=10)))

def apply_transform(transform):
    X = df.iloc[:,1:3]
    y = df.iloc[:,0]

    trf = ColumnTransformer([('log',FunctionTransformer(transform),['Fare'])],remainder='passthrough')

    X_trans = trf.fit_transform(X)

    clf = LogisticRegression()

    print("Accuracy",np.mean(cross_val_score(clf,X_trans,y,scoring='accuracy',cv=10)))

    plt.figure(figsize=(14,4))

    plt.subplot(121)
    stats.probplot(X['Fare'], dist="norm", plot=plt)
    plt.title('Fare Before Transform')

    plt.subplot(122)
    stats.probplot(X_trans[:,0], dist="norm", plot=plt)
    plt.title('Fare After Transform')

    plt.show()

apply_transform(np.sin)



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Generate 200 random X values between 0 and 10
X = np.random.uniform(0, 10, 200)

# Calculate Y using a quadratic relationship and add some noise
Y = -X**2 + 10*X + np.random.normal(0, 5, 200)

# Convert X and Y into a DataFrame for easier manipulation
df = pd.DataFrame({'X': X, 'Y': Y})

df.head()

df['X_squared'] = df['X']**2

df.head()

df_sorted = df.sort_values(by='X')

plt.scatter(df_sorted['X'], df_sorted['Y'], color='blue', label='Actual Data', alpha=0.6)

plt.scatter(df_sorted['X_squared'], df_sorted['Y'], color='blue', label='Actual Data', alpha=0.6)

# Linear model using X
linear_model = LinearRegression()
linear_model.fit(df[['X']], df['Y'])
df['Linear_Prediction'] = linear_model.predict(df[['X']])

# Linear model using X^2
df['X_squared'] = df['X']**2
squared_model = LinearRegression()
squared_model.fit(df[['X', 'X_squared']], df['Y'])
df['Squared_Prediction'] = squared_model.predict(df[['X', 'X_squared']])

# Sort the dataframe by X values for better plotting
df_sorted = df.sort_values(by='X')

# Plot the data and the models with sorted values
plt.figure(figsize=(10, 6))
plt.scatter(df_sorted['X'], df_sorted['Y'], color='blue', label='Actual Data', alpha=0.6)
plt.plot(df_sorted['X'], df_sorted['Linear_Prediction'], color='red', label='Linear Prediction')
plt.plot(df_sorted['X'], df_sorted['Squared_Prediction'], color='green', label='Squared Prediction')
plt.title('Comparison of Linear and Squared Models (Corrected)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

df.head()

df.skew()['crim']

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distplots without any transformation

for col in df.columns[0:-1]:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.kdeplot(df[col])
    txt = col + " " + str(df.skew()[col])
    plt.title(txt)
    plt.show()

# log on zn feature

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['zn'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.log1p(df['zn']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before log transform', skew(df['zn']))
print('skew after log transform', skew(np.log1p(df['zn'])))

# log on chas

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['chas'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.log1p(df['chas']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before reciprocal transform', skew(df['chas']))
print('skew after reciprocal transform', skew(np.log1p(df['chas'])))

# log on dis

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['dis'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.log1p(df['dis']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before log transform', skew(df['dis']))
print('skew after log transform', skew(np.log1p(df['dis'])))

# log on rad

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['rad'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.log1p(df['rad']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before log transform', skew(df['rad']))
print('skew after log transform', skew(np.log1p(df['rad'])))

# sqrt on indus

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['indus'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.sqrt(df['indus']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before sqrt transform', skew(df['indus']))
print('skew after sqrt transform', skew(np.sqrt(df['indus'])))

# sqrt on nox

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['nox'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.sqrt(df['nox']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before sqrt transform', skew(df['nox']))
print('skew after sqrt transform', skew(np.sqrt(df['nox'])))

# sqrt on rm

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['rm'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.sqrt(df['rm']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before sqrt transform', skew(df['rm']))
print('skew after sqrt transform', skew(np.sqrt(df['rm'])))

# sqrt on tax

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['tax'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.sqrt(df['tax']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before sqrt transform', skew(df['tax']))
print('skew after sqrt transform', skew(np.sqrt(df['tax'])))

# sqrt on lstat

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['lstat'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.sqrt(df['lstat']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before sqrt transform', skew(df['lstat']))
print('skew after sqrt transform', skew(np.sqrt(df['lstat'])))

# reciprocal on crim

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['crim'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.reciprocal(df['crim']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before reciprocal transform', skew(df['crim']))
print('skew after reciprocal transform', skew(np.reciprocal(df['crim'])))

# square on age

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['age'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.square(df['age']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before reciprocal transform', skew(df['age']))
print('skew after reciprocal transform', skew(np.square(df['age'])))

# square on ptratio

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['ptratio'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.square(df['ptratio']), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before reciprocal transform', skew(df['ptratio']))
print('skew after reciprocal transform', skew(np.square(df['ptratio'])))

-df['b']

# reflect log on b

# Setting up the matplotlib figure with two subplots
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

# Plotting the normal distribution
sns.kdeplot(df['b'], color='blue', ax=axs[0])

# Plotting the exponential distribution
sns.kdeplot(np.log1p(-df['b'] + 396.900000001), color='red', ax=axs[1])

plt.tight_layout()
plt.show()

from scipy.stats import skew

print('skew before reflect log transform', skew(df['b']))
print('skew after reflect log transform', skew(np.log1p(-df['b'] + 396.900000001)))

# log -> zn, chas, dis, rad
# sqrt -> indus, nox, rm, tax, lstat
# reciprocal -> crim
# square -> age, ptratio
# reflect shift log -> b

# baseline model

from sklearn.model_selection import train_test_split

# Define your features and target variable
X = df.drop('medv', axis=1)  # Features
y = df['medv']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_squared_error

y_pred = reg.predict(X_test)

print('r2 score', r2_score(y_test, y_pred))
print('mse', mean_squared_error(y_test, y_pred))

def reflect_shift_log_transform(x):

    return np.log1p(-x + 396.900000001)

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('log', FunctionTransformer(np.log1p), ['zn', 'chas', 'dis', 'rad']),
        ('sqrt', FunctionTransformer(np.sqrt), ['indus', 'nox', 'rm', 'tax', 'lstat']),
        ('reciprocal', FunctionTransformer(np.reciprocal), ['crim']),
        ('square', FunctionTransformer(np.square), ['age', 'ptratio']),
        ('reflect_shift_log', FunctionTransformer(reflect_shift_log_transform), ['b']),
    ],
    remainder='passthrough'  # Keeps columns not listed unchanged
)

X_train_trf = preprocessor.fit_transform(X_train)

X_test_trf = preprocessor.transform(X_test)

reg = LinearRegression()

reg.fit(X_train_trf, y_train)

y_pred = reg.predict(X_test_trf)

print('r2 score', r2_score(y_test, y_pred))
print('mse', mean_squared_error(y_test, y_pred))

"""### Box-Cox Transform"""

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')

X_train_transformed = pt.fit_transform(X_train+0.000001)
X_test_transformed = pt.transform(X_test+0.000001)

pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_})

reg = LinearRegression()

reg.fit(X_train_transformed, y_train)

y_pred = reg.predict(X_test_transformed)

print('r2 score', r2_score(y_test, y_pred))
print('mse', mean_squared_error(y_test, y_pred))

# Before and after comparision for Box-Cox Plot
X_train_transformed = pd.DataFrame(X_train_transformed,columns=X_train.columns)

for col in X_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.kdeplot(X_train[col])
    txt = col + " -> " + str(X_train[col].skew())
    plt.title(txt)

    plt.subplot(122)
    sns.kdeplot(X_train_transformed[col])
    txt1 = col + " -> " + str(X_train_transformed[col].skew())
    plt.title(txt1)

    plt.show()

"""### Yeo-Johnson"""

# Apply Yeo-Johnson transform

pt1 = PowerTransformer()

X_train_transformed2 = pt1.fit_transform(X_train)
X_test_transformed2 = pt1.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_transformed2,y_train)

y_pred3 = lr.predict(X_test_transformed2)

print(r2_score(y_test,y_pred3))

pd.DataFrame({'cols':X_train.columns,'Yeo_Johnson_lambdas':pt1.lambdas_})

# Before and after comparision for Box-Cox Plot
X_train_transformed2 = pd.DataFrame(X_train_transformed2 ,columns=X_train.columns)

for col in X_train_transformed2.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.kdeplot(X_train[col])
    txt = col + " -> " + str(X_train[col].skew())
    plt.title(txt)

    plt.subplot(122)
    sns.kdeplot(X_train_transformed2[col])
    txt1 = col + " -> " + str(X_train_transformed2[col].skew())
    plt.title(txt1)

    plt.show()

# Side by side Lambdas
pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_,'Yeo_Johnson_lambdas':pt1.lambdas_})

