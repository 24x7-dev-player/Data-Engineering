import pandas as pd
import random
L = []
for i in range(10000):
  a = random.randint(1,6)
  b = random.randint(1,6)
  L.append(a + b)
len(L)
L[:5]
s = (pd.Series(L).value_counts()/pd.Series(L).value_counts().sum()).sort_index()

import numpy as np
np.cumsum(s)
s.plot(kind='bar')
np.cumsum(s).plot(kind='bar')

"""# Parametric Density Estimation"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
sample = normal(loc=50, scale=5,size=1000)
sample.mean()

# plot histogram to understand the distribution of data
plt.hist(sample,bins=10)

# calculate sample mean and sample std dev
sample_mean = sample.mean()
sample_std = sample.std()

# fit the distribution with the above parameters

from scipy.stats import norm
dist = norm(60, 12)

values = np.linspace(sample.min(),sample.max(),100)

sample.max()

probabilities = [dist.pdf(value) for value in values]

# plot the histogram and pdf
plt.hist(sample,bins=10,density=True)
plt.plot(values,probabilities)

import seaborn as sns
sns.distplot(sample)

"""# KDE"""

# generate a sample
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = np.hstack((sample1, sample2))

sample

# plot histogram bins=50
plt.hist(sample,bins=50)

from sklearn.neighbors import KernelDensity

model = KernelDensity(bandwidth=5, kernel='gaussian')

# convert data to a 2D array
sample = sample.reshape((len(sample), 1))

model.fit(sample)

values = np.linspace(sample.min(),sample.max(),100)
values = values.reshape((len(values), 1))

probabilities = model.score_samples(values)
probabilities = np.exp(probabilities)

"""`score_samples(values)` returns the log-density estimate of the input samples values. This is because the `score_samples()` method of the KernelDensity class returns the logarithm of the probability density estimate rather than the actual probability density estimate."""

plt.hist(sample, bins=50, density=True)
plt.plot(values[:], probabilities)
plt.show()

sns.kdeplot(sample.reshape(1000),bw_adjust=0.02)

import seaborn as sns

df = sns.load_dataset('iris')

df.head()

sns.kdeplot(data=df,x='sepal_length',hue='species')

sns.kdeplot(data=df,x='sepal_width',hue='species')

sns.kdeplot(data=df,x='petal_length',hue='species')

sns.kdeplot(data=df,x='petal_width',hue='species')



sns.kdeplot(df['petal_width'],hue=df['species'])
sns.ecdfplot(data=df,x='petal_width',hue='species')

titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

titanic.head()

# code here
sns.kdeplot(data=titanic,x='Age',hue='Sex')

sns.jointplot(data=df, x="petal_length", y="sepal_length", kind="kde",fill=True,cbar=True)

sns.kdeplot(titanic['Age'])

titanic['Age'].mean()

x = (titanic['Age'] - titanic['Age'].mean())/titanic['Age'].std()

sns.kdeplot(x)

x.mean()

x.std()

titanic['Age'].skew()

titanic['Age'].mean() + 3*titanic['Age'].std()

titanic['Age'].mean() - 3*titanic['Age'].std()

titanic[titanic['Age'] > 73]

titanic['Age'].max()

