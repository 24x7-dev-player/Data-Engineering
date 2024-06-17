import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('cars.csv')

df.head()

df['owner'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(
                                                      df.drop(columns=['selling_price']),
                                                      df['selling_price'],
                                                      test_size=0.2,
                                                      random_state=42
                                                    )

X_train.head()

"""### The Hard Way!"""

# apply ordinal encoder to owner
oe = OrdinalEncoder(categories=[['Test Drive Car', 'Fourth & Above Owner', 'Third Owner', 'Second Owner', 'First Owner']])

X_train_owner = oe.fit_transform(X_train.loc[:,['owner']])
X_test_owner = oe.transform(X_test.loc[:,['owner']])

# convert to df
X_train_owner_df = pd.DataFrame(X_train_owner,columns=oe.get_feature_names_out())
X_test_owner_df = pd.DataFrame(X_test_owner,columns=oe.get_feature_names_out())

X_train_owner_df.head()

# apply ohe to brand and fuel
ohe = OneHotEncoder(sparse_output=False)

X_train_brand_fuel = ohe.fit_transform(X_train[['brand','fuel']])
X_test_brand_fuel = ohe.transform(X_test[['brand','fuel']])

# converting to dataframe
X_train_brand_fuel_df = pd.DataFrame(X_train_brand_fuel, columns=ohe.get_feature_names_out())
X_test_brand_fuel_df = pd.DataFrame(X_test_brand_fuel, columns=ohe.get_feature_names_out())

X_train_brand_fuel_df.head()

X_train.head()

X_train_rem = X_train.drop(columns=['brand','fuel','owner'],inplace=True)
X_test_rem = X_test.drop(columns=['brand','fuel','owner'],inplace=True)

X_train = pd.concat([X_train_rem, X_train_owner_df, X_train_brand_fuel_df],axis=1)
X_test = pd.concat([X_test_rem, X_test_owner_df, X_test_brand_fuel_df],axis=1)

X_train.head()

"""### The Easy Way!"""

from sklearn.compose import ColumnTransformer

df = pd.read_csv('cars.csv')

X_train, X_test, y_train, y_test = train_test_split(
                                                      df.drop(columns=['selling_price']),
                                                      df['selling_price'],
                                                      test_size=0.2,
                                                      random_state=42
                                                    )

X_train.head()

transformer = ColumnTransformer(
    [
        ("ordinal", OrdinalEncoder(categories=[['Test Drive Car', 'Fourth & Above Owner', 'Third Owner', 'Second Owner', 'First Owner']]), ['owner']),
        ("onehot", OneHotEncoder(sparse_output=False), ['brand', 'fuel'])
    ],
    remainder='passthrough'
)

# setting to get a pandas df
transformer.set_output(transform='pandas')

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

transformer.set_output(transform='pandas')

transformer.fit_transform(X_train)

transformer.feature_names_in_

transformer.get_feature_names_out()

transformer.n_features_in_

transformer.transformers_

transformer.output_indices_

"""### Sklearn Pipeline"""

df = pd.read_csv('cars.csv')
df.head()

df.shape

import numpy as np

np.random.seed(42)
missing_km_indices = np.random.choice(df.index, size=int(0.05*len(df)), replace=False)
df.loc[missing_km_indices, 'km_driven'] = np.nan

# Introduce missing values in 'owner' column (1% missing values)
missing_owner_indices = np.random.choice(df.index, size=int(0.01*len(df)), replace=False)
df.loc[missing_owner_indices, 'owner'] = np.nan

df.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(
                                                      df.drop(columns=['selling_price']),
                                                      df['selling_price'],
                                                      test_size=0.2,
                                                      random_state=42
                                                    )

X_train.head()

X_train.info()

# Plan of Attack

# Missing value imputation
# Encoding Categorical Variables
# Scaling
# Feature Selection
# Model building
# Prediction

df['owner'].value_counts()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# imputation transformer
trf1 = ColumnTransformer([
    ('impute_km_driven',SimpleImputer(),[1]),
    ('impute_owner',SimpleImputer(strategy='most_frequent'),[3])
],remainder='passthrough')

# encoding categorical variables
trf2 = ColumnTransformer(
    [
        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [3]),
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), [0,2])
    ],
    remainder='passthrough'
)

# Scaling
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,38))
])

a = [1,2,3,4,5]
x = slice(0,5)
a[x]

# Feature selection
trf4 = SelectKBest(score_func=chi2,k=10)

# train the model
trf5 = RandomForestRegressor()

from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer',trf1),
    ('encoder',trf2),
    ('scaler',trf3),
    ('fselector',trf4),
    ('model',trf5)
])

pipe.fit(X_train, y_train)

pipe.feature_names_in_

pipe.named_steps

pipe.named_steps['scaler'].transformers_[0][1].data_max_

pipe.predict(X_test)[10:40]

# Predict
pipe.predict(np.array(['Maruti',100000.0,'Diesel','First Owner']).reshape(1,4))

"""### Cross Validation"""

# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

"""### Hyperparameter Tuning"""

# gridsearchcv
params = {
    'model__max_depth':[1,2,3,4,5,None]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

grid.best_score_

grid.best_params_

"""### Export the Pipeline"""

# export
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))