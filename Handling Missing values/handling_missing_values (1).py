import pandas as pd

# Load your dataset
# Assuming your dataset is in a CSV file named 'dataset.csv'
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Check for missing values in each column
missing_values_count = df.isnull().sum()

# Print the number of missing values for each column
print("Number of missing values in each column:")
print(missing_values_count)

# If you want to see the percentage of missing values in each column
missing_values_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_values_percentage)

# To get a quick overview of total missing values in the dataset
total_missing_values = df.isnull().sum().sum()
print(f"\nTotal missing values in the dataset: {total_missing_values}")

# Check for any rows that have missing values
rows_with_missing_values = df[df.isnull().any(axis=1)]
print("\nRows with missing values:")
rows_with_missing_values

"""### Missingo for Missing Values"""

!pip install missingno

df = pd.read_csv('https://raw.githubusercontent.com/andymcdgeo/Andys_YouTube_Notebooks/main/Data/xeek_subset2.csv')
df.head()

df.shape

df.info()

import missingno as msno
import matplotlib.pyplot as plt

# Bar plot to visualize missing values
msno.bar(df)

# Matrix chart to visualize missing values
msno.matrix(df)
plt.show()

# Heatmap to visualize the correlation of missingness between columns
msno.heatmap(df)
plt.show()

# Dendrogram to visualize the hierarchical clustering of missing values
msno.dendrogram(df)
plt.show()

import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random data
data = np.random.randn(100, 3)  # 100 rows, 3 columns

# Artificially introduce missing values completely at random in the first column
missing_indices = np.random.choice(np.arange(100), size=20, replace=False)
data[missing_indices, 0] = np.nan

# Create a DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

df.head()

df.shape

# Split the data into two groups
missing_data = df[df['Feature1'].isnull()]
not_missing_data = df[~df['Feature1'].isnull()]

# Calculate the means
means_missing = missing_data.mean()
means_not_missing = not_missing_data.mean()

# Print the means for comparison
print("Means with missing data:\n", means_missing)
print("\nMeans without missing data:\n", means_not_missing)

not_missing_data.shape

import numpy as np
from scipy import stats

# Compare means of another feature 'feature2' between groups
t_stat, p_val = stats.ttest_ind(missing_data['Feature2'], not_missing_data['Feature2'], nan_policy='omit')

print(f"T-statistic: {t_stat}, P-value: {p_val}")

import missingno as msno
import matplotlib.pyplot as plt

# Matrix plot to visualize missing values
msno.matrix(df)
plt.show()

"""### How to identify MAR"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 100
f2 = np.random.randn(n_samples)  # Feature 2: Normal distribution
f3 = np.random.rand(n_samples) * 100  # Feature 3: Uniform distribution between 0 and 100

# Generate Feature 1 with a dependency on Feature 2 for missingness
f1 = np.random.randn(n_samples) * 50  # Initial Feature 1: Normal distribution, scaled
# Introduce missing values in f1 based on f2; higher values of f2 are more likely to result in missing f1 values
missing_probability = (f2 - f2.min()) / (f2.max() - f2.min())  # Normalize f2 to get probabilities
f1[missing_probability > 0.8] = np.nan  # Set f1 to NaN where missing_probability > 0.8

# Create DataFrame
df = pd.DataFrame({'Feature1': f1, 'Feature2': f2, 'Feature3': f3})

df.head()

df.isnull().sum()

# Matrix plot to visualize the missing data
msno.matrix(df)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create a binary indicator for missingness in Feature1
df['Feature1_missing'] = df['Feature1'].isnull().astype(int)

# Prepare the independent variables (X) and the target variable (y)
X = df[['Feature2', 'Feature3']]  # or just df[['Feature2']] if focusing on Feature2
y = df['Feature1_missing']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

model.coef_

import matplotlib.pyplot as plt
import seaborn as sns

# Split the DataFrame into two groups based on missingness in Feature1
group_with_missing = df[df['Feature1'].isnull()]
group_without_missing = df[~df['Feature1'].isnull()]

# Compare distributions of Feature2 and Feature3 across the two groups
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plotting the distribution of Feature2 for both groups
sns.kdeplot(group_with_missing['Feature2'], color='red', label='With Missing Feature1', ax=axes[0])
sns.kdeplot(group_without_missing['Feature2'], color='blue', label='Without Missing Feature1', ax=axes[0])
axes[0].set_title('Distribution of Feature2')
axes[0].legend()

# Plotting the distribution of Feature3 for both groups
sns.kdeplot(group_with_missing['Feature3'], color='red', label='With Missing Feature1', ax=axes[1])
sns.kdeplot(group_without_missing['Feature3'], color='blue', label='Without Missing Feature1', ax=axes[1])
axes[1].set_title('Distribution of Feature3')
axes[1].legend()

plt.tight_layout()
plt.show()

"""### Listwise deletion (CCA)"""

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day35-complete-case-analysis/data_science_job.csv')
df.head()

df.isnull().mean()*100

df = df[['city_development_index', 'enrolled_university',	'education_level',	'experience',	'training_hours']]
df.head()

# CCA applied
new_df = df.dropna()
df.shape, new_df.shape

len(new_df) / len(df)

new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].plot.density(color='red')

# data after cca
new_df['training_hours'].plot.density(color='green')

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['city_development_index'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].plot.density(color='red')

# data after cca
new_df['city_development_index'].plot.density(color='green')

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['experience'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['experience'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['experience'].plot.density(color='red')

# data after cca
new_df['experience'].plot.density(color='green')

temp = pd.concat([
            # percentage of observations per category, original data
            df['enrolled_university'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['enrolled_university'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp

temp = pd.concat([
            # percentage of observations per category, original data
            df['education_level'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['education_level'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp

"""### Missing Indicator"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import MissingIndicator,SimpleImputer

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',usecols=['Age','Fare', 'Cabin','Survived'])
df.head()

df.info()

X = df.drop(columns=['Survived', 'Cabin'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf,y_train)

y_pred = clf.predict(X_test_trf)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer(
    transformers=[
        ('age_imputer', SimpleImputer(strategy='mean'), ['Age']),
        ('cabin_indicator', MissingIndicator(), ['Cabin'])
    ],
    remainder='passthrough'  # This specifies that columns not explicitly selected should be passed through without transformation
)

X_train_new = column_transformer.fit_transform(X_train)
X_test_new = column_transformer.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_new,y_train)

y_pred = clf.predict(X_test_new)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

"""### Using Mean and Median Imputation"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day36-imputing-numerical-data/titanic_toy.csv')
df.head()

df.isnull().mean()

import missingno as msno
import matplotlib.pyplot as plt

# Matrix plot to visualize missing values
msno.matrix(df)
plt.show()

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

mean_age = X_train['Age'].mean()
median_age = X_train['Age'].median()

mean_fare = X_train['Fare'].mean()
median_fare = X_train['Fare'].median()

X_train['Age_median'] = X_train['Age'].fillna(median_age)
X_train['Age_mean'] = X_train['Age'].fillna(mean_age)

X_train['Fare_median'] = X_train['Fare'].fillna(median_fare)
X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare)

X_train.sample(5)

print('Original Age variable variance: ', X_train['Age'].var())
print('Age Variance after median imputation: ', X_train['Age_median'].var())
print('Age Variance after mean imputation: ', X_train['Age_mean'].var())
print("-"*70)
print('Original Fare variable variance: ', X_train['Fare'].var())
print('Fare Variance after median imputation: ', X_train['Fare_median'].var())
print('Fare Variance after mean imputation: ', X_train['Fare_mean'].var())

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['Age'].plot(kind='kde', ax=ax)

# variable imputed with the median
X_train['Age_median'].plot(kind='kde', ax=ax, color='red')

# variable imputed with the mean
X_train['Age_mean'].plot(kind='kde', ax=ax, color='green')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['Fare'].plot(kind='kde', ax=ax)

# variable imputed with the median
X_train['Fare_median'].plot(kind='kde', ax=ax, color='red')

# variable imputed with the mean
X_train['Fare_mean'].plot(kind='kde', ax=ax, color='green')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

X_train.corr()

X_train[['Age', 'Age_median', 'Age_mean']].boxplot()

X_train[['Fare', 'Fare_median', 'Fare_mean']].boxplot()

"""### using sklearn"""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

imputer1 = SimpleImputer(strategy='most_frequent')
imputer2 = SimpleImputer(strategy='mean')

trf = ColumnTransformer([
    ('imputer1',imputer1,['Age']),
    ('imputer2',imputer2,['Fare'])
],remainder='passthrough')

trf.fit(X_train)

trf.named_transformers_['imputer1'].statistics_

trf.named_transformers_['imputer2'].statistics_

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

X_train

