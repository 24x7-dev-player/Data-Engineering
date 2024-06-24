import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')[['Survived','Pclass','Age','Fare']]


df.head()

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Imputation
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=0), max_iter=10, random_state=0)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert imputed data back to DataFrame (optional, for clarity)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

X_train_imputed

# Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train_imputed, y_train)

# Predict on the test set
y_pred = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

import pandas as pd

# URL of the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Load the dataset
df = pd.read_csv(url)

# Display the first few rows of the DataFrame to verify it loaded correctly
df.head()

# Drop 'PassengerId', 'Name', and 'Ticket' columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Display the first few rows of the DataFrame to verify the columns have been dropped
df.head()

from sklearn.model_selection import train_test_split

# Separate the features and the target variable
X = df.drop('Survived', axis=1)  # Features (all columns except 'Survived')
y = df['Survived']  # Target variable

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the split
X_train.head()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Since 'Embarked' is categorical and we're filling missing values with the most frequent category,
# it's useful to apply OneHotEncoding to it as well after imputation.
# We can achieve this by setting up a pipeline for 'Embarked'.
embarked_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder())
])

# Update the transformations to use the pipeline for 'Embarked'
transformations = ColumnTransformer(transformers=[
    ('ohe_sex', OneHotEncoder(), ['Sex']),
    ('impute_age', SimpleImputer(strategy='mean'), ['Age']),
    ('missing_indicator', MissingIndicator(), ['Cabin']),
    ('embarked_pipeline', embarked_pipeline, ['Embarked'])
])

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', transformations),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Now you can use the pipeline to make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model, e.g., by calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

from sklearn.model_selection import train_test_split, GridSearchCV

# Parameters of the pipeline to tune
param_grid = {
    'preprocessor__impute_age': [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), SimpleImputer(strategy='constant', fill_value=0)],
    'preprocessor__embarked_pipeline': [Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder())]),
                                        Pipeline(steps=[('impute', SimpleImputer(strategy='constant', fill_value='S')), ('ohe', OneHotEncoder())])]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Best parameter set found
print("Best parameters found:\n", grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with best parameters: {accuracy:.4f}")

"""### Having Multiple Approaches"""

import pandas as pd

# URL of the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Load the dataset
df = pd.read_csv(url).drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'])

from sklearn.model_selection import train_test_split

# Separate the features and the target variable
X = df.drop('Survived', axis=1)  # Features (all columns except 'Survived')
y = df['Survived']  # Target variable

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the split
X_train.head()

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer


# Update the transformations to use the pipeline for 'Embarked'
approach_1_preprocessor = ColumnTransformer(transformers=[
    ('ohe_sex', OneHotEncoder(), ['Sex']),
    ('impute_age', SimpleImputer(strategy='mean'), ['Age']),
    ('missing_indicator', MissingIndicator(), ['Cabin'])
], remainder='passthrough')



approach_2_preprocessor = ColumnTransformer(transformers=[
    ('ohe_sex', OneHotEncoder(), ['Sex']),
    ('knn_impute', KNNImputer(), ['Age']),
    ('missing_indicator', MissingIndicator(), ['Cabin'])
], remainder='passthrough')


# Approach 3: IterativeImputer for Age and Embarked, OHE for Sex, Missing Indicator for Cabin
approach_3_preprocessor = ColumnTransformer(transformers=[
    ('ohe_sex', OneHotEncoder(), ['Sex']),
    ('iterative_impute', IterativeImputer(random_state=0), ['Age']),
    ('missing_indicator', MissingIndicator(), ['Cabin'])
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', None),  # Placeholder
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameters of the pipeline to tune, including the entire preprocessor component
param_grid = {
    'preprocessor': [approach_3_preprocessor, approach_2_preprocessor, approach_1_preprocessor]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Best parameter set found
print("Best parameters found:\n", grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with best parameters: {accuracy:.4f}")

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


param_grid = [
    {
        'preprocessor': [approach_1_preprocessor],  # Approach 1
        'preprocessor__impute_age__strategy': ['mean', 'median', 'constant']  # Tuning SimpleImputer within Approach 1
    },
    {
        'preprocessor': [approach_2_preprocessor],  # Approach 2
        'preprocessor__knn_impute__n_neighbors': [3, 5, 7],  # Tuning KNNImputer within Approach 2
        'preprocessor__knn_impute__weights': ['uniform', 'distance']  # Additional KNNImputer parameter
    },
    {
        'preprocessor': [approach_3_preprocessor],  # Approach 3
        'preprocessor__iterative_impute__max_iter': [10, 20],  # Tuning IterativeImputer within Approach 3
        'preprocessor__iterative_impute__imputation_order': ['ascending', 'descending', 'roman', 'arabic'],  # Additional IterativeImputer parameter
        'preprocessor__iterative_impute__estimator': [BayesianRidge(), ExtraTreesRegressor(n_estimators=50, random_state=0), RandomForestRegressor(random_state=0)]
    }
]

# Set up the GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Best parameter set found
print("Best parameters found:\n", grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with best parameters: {accuracy:.4f}")

"""# Comparison

| Feature/Method             | Mean Imputation | Median Imputation | Most Frequent Imputation | Constant Value Imputation | KNN Imputer | Missing Indicator | Iterative Imputer |
|----------------------------|-----------------|-------------------|--------------------------|---------------------------|-------------|-------------------|-------------------|
| **Suitable Data Types**    | Numeric only    | Numeric only      | Numeric and Categorical  | Numeric and Categorical   | Numeric     | Numeric and Categorical | Numeric primarily; Categorical with preprocessing |
| **Use Case**               | Simple cases, quick baseline | Non-normally distributed data | Categorical or when a mode is clear | When a placeholder is needed | Data with meaningful neighbor relationships | To flag missingness as a feature | Complex relationships, multiple variables with missing data |
| **Advantages**             | Easy to implement, quick | Robust to outliers | Good for categorical data | Flexibility in handling missing data | Captures local data structure | Directly models the impact of missingness | Utilizes inter-feature relationships, flexible estimator choice |
| **Disadvantages**          | Can distort distribution, reduce variance | Can distort distribution if not normally distributed | May not reflect underlying data complexity | May introduce artificial variance | Computationally intensive, sensitive to outliers | Increases feature space | Computationally expensive, risk of overfitting |
| **Assumes Data Pattern**   | MCAR            | MCAR              | MCAR/MAR                 | MNAR             | MAR    | MNAR     | MAR |
| **Complexity**             | Low             | Low               | Low                      | Low                       | High        | Low               | High |
| **Handling Missingness**   | Directly fills missing values | Directly fills missing values | Directly fills missing values | Directly fills missing values with a constant | Fills based on nearest neighbors | Creates binary indicators for missingness | Models each feature with missing values as a function of others |
| **Impact on Distribution** | Can distort the original distribution by affecting mean and reducing variance | Less impact on distribution for skewed data but can still alter original distribution | May not reflect the true distribution, especially if the mode is not representative of the data | No impact on the original distribution of the variable, but introduces a distinct category | Tries to maintain the local structure of the data, less distortion if neighbors are representative | No direct impact on the original distribution of the variable, but adds new binary features | Attempts to preserve relationships and distributions by using other features, but effectiveness varies with the underlying estimator |
| **Model Performance**      | May decrease performance if mean is not representative | Better for skewed data, but similar issues as mean imputation | Good for nominal categorical data with a clear mode | Useful when a distinct category for missing values is meaningful | Can improve performance if the dataset has a meaningful structure that neighbors can capture | Useful for models that can leverage the presence of missingness as an informative signal | Can improve model performance by leveraging inter-feature correlations, but depends on estimator selection |
| **Computational Cost**     | Low             | Low               | Low                      | Low                       | High        | Low               | High (multiple iterations and model fitting involved) |
| **Best Use Cases**         | Quick baseline models or when data is normally distributed and missing completely at random (MCAR) | Data with outliers or non-normal distribution, MCAR | Categorical data or when a clear majority category exists, MCAR | Situations where missingness might represent a distinct category itself | Data with rich feature interactions or when the local neighborhood can accurately predict missing values | Models where missingness itself is informative, regardless of the missing values | Complex datasets with multiple features having interdependencies, especially when data is not missing completely at random (MAR) |
| **Scalability**            | Highly scalable | Highly scalable   | Highly scalable           | Highly scalable            | Scalability issues with large datasets due to the need to compute distances between points | Highly scalable | Less scalable due to iterative nature and the need for multiple model fittings |
| **Risk of Bias**           | Introduces bias if the mean is not representative of the missing values | Lower risk of bias compared to mean imputation but still present | Risk of bias if the mode does not represent missing values well | Risk of introducing artificial variance if the constant does not represent the missing context well | Lower risk of bias if KNN can accurately capture the data structure, but sensitive to outliers | Minimal direct bias in imputation, but model interpretation complexity increases | Risk of overfitting or underfitting depending on the complexity of the estimator and the accuracy of the initial imputation |
| **Special Considerations** | Simple and fast, but may not be suitable for datasets with complex relationships or non-random missingness patterns | Similar to mean imputation but more robust to outliers | Useful for nominal data; requires careful consideration for ordinal or interval data | Flexible in handling different missingness contexts but requires thoughtful choice of the constant value | Requires careful tuning of `n_neighbors` and distance metric; performance may vary with data dimensionality | Can significantly increase the feature space; effectiveness depends on the downstream model's capacity | Requires selection of an appropriate estimator; computational demand and convergence criteria need careful management |
"""

