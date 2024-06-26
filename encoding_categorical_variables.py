import numpy as np
import pandas as pd

df = pd.read_csv('/content/customer.csv')

df.head()

"""### 1. Ordinal Encoding"""

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2], df.iloc[:,-1], test_size=0.2)

X_train

# specify order
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])

X_train = oe.fit_transform(X_train)
X_test = oe.transform(X_test)

X_train

oe.categories_

oe.feature_names_in_

oe.n_features_in_

oe.inverse_transform(np.array([0,2]).reshape(1,2))

oe.get_feature_names_out()

# handle unknown
oe.transform(np.array(['Poor','college']).reshape(1,2))

# set unknown value
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2], df.iloc[:,-1], test_size=0.2)
X_train = oe.fit_transform(X_train)
oe.transform(np.array(['Poor','college']).reshape(1,2))

# handling infrequent categories
X = np.array([['dog'] * 5 + ['cat'] * 20 + ['rabbit'] * 10 +['snake'] * 3 + ['horse'] * 2], dtype=object).T
X

enc = OrdinalEncoder(max_categories=3).fit(X)

enc.infrequent_categories_

enc.transform(np.array([['cat','rabbit','snake','dog']]).reshape(4,1))

enc = OrdinalEncoder(min_frequency=4).fit(X)
enc.infrequent_categories_

enc.transform(np.array([['cat','rabbit','snake','dog','horse']]).reshape(5,1))

# update sklearn if the above command doesn't work
!pip install --upgrade scikit-learn==1.3.2

# handling missing data

# Example categorical data with missing values
data = [['Cat'], [np.nan], ['Dog'], ['Fish'], [np.nan]]

# Setting encoded_missing_value to -1, indicating we want missing values to be encoded as -1
encoder = OrdinalEncoder(encoded_missing_value=-1)

encoded_data = encoder.fit_transform(data)

print(encoded_data)

"""### LabelEncoder"""

df.head()

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2], df.iloc[:,-1], test_size=0.2)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

le.classes_

le.inverse_transform(np.array([1,1,0]))

"""### 2. OneHotEncoder"""

cars = pd.read_csv('cars.csv')
cars.head()

cars.isnull().sum()

X = cars.iloc[:,0:2]
y = cars.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train['fuel'].nunique()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype=np.int32)

ohe.fit_transform(X_train)

X_train = ohe.fit_transform(X_train).toarray()
X_train

X_train.shape

ohe.categories_

ohe.feature_names_in_

ohe.n_features_in_

ohe.get_feature_names_out()

pd.DataFrame(ohe.fit_transform(X_train),columns=ohe.get_feature_names_out())

# form dataframe again if required

ohe.inverse_transform(np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.]).reshape(1,34))

# show sparse_array false

# show dtype

# drop
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ohe = OneHotEncoder(drop='first',sparse_output=False)
ohe.fit_transform(X_train).shape

ohe.drop_idx_

# handling rare categories
X_train['brand'].value_counts()

cars['fuel'].value_counts()

# using min frequency

ohe = OneHotEncoder(sparse_output=False, min_frequency=100)
ohe.fit_transform(X_train).shape

ohe.get_feature_names_out()

# using max_categories
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=15)
ohe.fit_transform(X_train).shape

ohe.get_feature_names_out()

# how to handle unknown category
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ohe = OneHotEncoder(drop='first',sparse_output=False)
ohe.fit_transform(X_train)

ohe.transform(np.array(['local','Petrol']).reshape(1,2))

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit_transform(X_train)

ohe.transform(np.array(['local','Petrol']).reshape(1,2))

ohe.inverse_transform(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1.]).reshape(1,36))

"""### LabelBinarizer"""

from sklearn.preprocessing import LabelBinarizer

# Sample target variable for a multi-class classification problem
y = ['cat', 'dog', 'fish', 'dog', 'cat']

# Initialize the LabelBinarizer
lb = LabelBinarizer()

# Fit and transform the target variable
y_binarized = lb.fit_transform(y)

print("Binarized labels:\n", y_binarized)

# Inverse transform to recover original labels
y_original = lb.inverse_transform(y_binarized)

print("Original labels:\n", y_original)

from sklearn.preprocessing import MultiLabelBinarizer

# Example multi-label data
y = [('red', 'blue'), ('blue', 'green'), ('green',), ('red',)]

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform the data to binary matrix format
Y = mlb.fit_transform(y)

print("Binary matrix:\n", Y)
print("Class labels:", mlb.classes_)

# Inverse transform to recover original labels
y_inv = mlb.inverse_transform(Y)
print("Inverse transformed labels:", y_inv)

"""### 3. Count Encoder/Frequency Encoder"""

!pip install category_encoders

# dataset generation
import pandas as pd
import numpy as np
import category_encoders as ce

# Simulating a dataset
data = {
    'Age': np.random.randint(20, 60, size=100).astype(float),  # Random ages between 20 and 60
    'State': np.random.choice(['Karnataka', 'Tamil Nadu', 'Maharashtra', 'Delhi', 'Telangana'], size=100),
    'Education': np.random.choice(['High School', 'UG', 'PG'], size=100),
    'Package': np.random.rand(100) * 100  # Random package values for demonstration
}

# Introducing missing values in 'Age' column (5%)
np.random.seed(0)  # For reproducibility
missing_indices = np.random.choice(data['Age'].shape[0], replace=False, size=int(data['Age'].shape[0] * 0.05))
data['Age'][missing_indices] = np.nan

df = pd.DataFrame(data)

df.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Package']), df['Package'], test_size=0.2, random_state=42)

X_train.head()

X_train['State'].value_counts()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import sklearn

class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.count_map = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        for col in self.columns:
            self.count_map[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.count_map[col]).fillna(0)
        return X

preprocessor = ColumnTransformer(
    transformers=[
        ('age_missing', SimpleImputer(strategy='mean'), ['Age']),
        ('cat_state', CountEncoder(), ['State']),
        ('education_ordinal', OrdinalEncoder(), ['Education'])
    ])

sklearn.set_config(transform_output="pandas")

preprocessor.fit_transform(X_train)

# using category encoders
from category_encoders.count import CountEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('age_missing', SimpleImputer(strategy='mean'), ['Age']),
        ('cat_state', CountEncoder(normalize=True), ['State']),
        ('education_ordinal', OrdinalEncoder(), ['Education'])
    ])
sklearn.set_config(transform_output="pandas")

preprocessor.fit_transform(X_train)

# frequency encoding

# parameters
import pandas as pd
import numpy as np
import category_encoders as ce

# Simulating a dataset
np.random.seed(42)  # For reproducibility
data = {
    'State': np.random.choice(['Karnataka', 'Tamil Nadu', 'Maharashtra', 'Delhi', 'Telangana', np.NaN], size=100),
    'Education': np.random.choice(['High School', 'UG', 'PG', np.NaN], size=100)
}
df = pd.DataFrame(data)

df.head(25)

df.isnull().sum()

# Initialize the CountEncoder with various parameters
encoder = ce.CountEncoder(
    cols=['State', 'Education'],  # Specify columns to encode. None would automatically select categorical columns.
    handle_missing='error',  # Treat NaNs as a countable category
    handle_unknown='error',  # Treat unknown categories as NaNs (if seen during transform but not in fit)
)

# Fit and transform the dataset
encoder.fit_transform(df)

#print(encoded_df.head(25))

encoder.mapping

new_data = pd.DataFrame({'State': ['Bihar'], 'Education': ['UG']})

encoder.transform(new_data)

np.random.seed(0)  # For reproducibility
data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', np.nan], size=100, p=[0.3, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05]),
    'Value': np.random.rand(100)
}

df = pd.DataFrame(data)

df.sample(10)

df['Category'].value_counts()

encoder = ce.CountEncoder(
    cols=['Category'],
    min_group_size=10,  # Groups with counts less than 5 will be combined
    min_group_name='salman',  # Use default naming for combined minimum groups
)

# Fit and transform the dataset
encoded_df = encoder.fit_transform(df['Category'])

# Display the original and encoded data for comparison
df['Encoded'] = encoded_df
print(df.head(20))

encoder.mapping

"""### Binary Encoder"""

import pandas as pd
import category_encoders as ce

# Sample dataset
data = {
    'Item': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8'],
    'Fruit': ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry', 'Fig', 'Grape', 'Honeydew']
}
df = pd.DataFrame(data)

df

# Initialize the Binary Encoder
encoder = ce.BinaryEncoder(cols=['Fruit'], return_df=True)

# Fit and transform the data
df_encoded = encoder.fit_transform(df)

# Display the original and encoded data
print(df_encoded)

"""### Target Encoder"""

# using category_encoder

import pandas as pd
import category_encoders as ce

# Sample data
data = {
    'Feature': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Target': [1, 0, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separating the feature and target columns
X = df.drop('Target', axis=1)
y = df['Target']

# Initialize the TargetEncoder
encoder = ce.TargetEncoder(cols=['Feature'])

# Fit the encoder using the feature data and target variable
encoder.fit(X, y)

# Transform the data
encoded = encoder.transform(X)

# Show the original and encoded data
print(pd.concat([df, encoded], axis=1))

encoder.mapping

!pip install --upgrade scikit-learn==1.4.0

# using sklearn
import pandas as pd
from sklearn.preprocessing import TargetEncoder

# Sample data
data = {
    'Feature': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Target': [1, 0, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separating the feature and target columns
X = df.drop('Target', axis=1)
y = df['Target']

# Initialize the TargetEncoder
encoder = TargetEncoder(smooth=0.0)

# Fit the encoder using the feature data and target variable
encoder.fit(X, y)

# Transform the data
encoded = encoder.transform(X)

encoded

"""### Weight of Evidence"""

!pip install category_encoders

import pandas as pd
import category_encoders as ce

# Example dataset
data = {
    'Feature': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Target': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Define the features and target
X = df[['Feature']]
y = df['Target']

# Initialize and fit the TargetEncoder
encoder = ce.WOEEncoder(cols=['Feature'])
X_encoded = encoder.fit_transform(X, y)

# Display the original and encoded data
df['Feature_Encoded'] = X_encoded
print(df)

