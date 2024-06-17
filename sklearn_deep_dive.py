from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.utils import check_X_y

"""### Custom Estimator"""

class MostFrequentClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.most_frequent_ = None

    def fit(self, X, y):

        # Validate input X and target vector y
        X, y = check_X_y(X, y)

        # Ensure y is 1D
        y = np.ravel(y)

        # Manually compute the most frequent class
        unique_classes, counts = np.unique(y, return_counts=True)
        self.most_frequent_ = unique_classes[np.argmax(counts)]

        return self

    def predict(self, X):
        if self.most_frequent_ is None:
            raise ValueError("This classifier instance is not fitted yet.")
        # Predict the most frequent class for each input sample
        return np.full(shape=(X.shape[0],), fill_value=self.most_frequent_)

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and fit the custom estimator
classifier = MostFrequentClassClassifier()
classifier.fit(X_train, y_train)

# Make predictions
#predictions = classifier.predict(X_test)

# Evaluate the custom estimator
print(f"Predicted class for all test instances: {predictions[0]}")

classifier.most_frequent_

from sklearn.model_selection import cross_val_score

cross_val_score(classifier, X_train, y_train)

"""### Scoing function"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

class MostFrequentClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.most_frequent_ = None

    def fit(self, X, y):
        # Ensure y is 1D
        y = np.ravel(y)

        # Compute the most frequent class
        unique_classes, counts = np.unique(y, return_counts=True)
        self.most_frequent_ = unique_classes[np.argmax(counts)]
        return self

    def predict(self, X):
        if self.most_frequent_ is None:
            raise ValueError("This classifier instance is not fitted yet.")
        # Predict the most frequent class for each input sample
        return np.full(shape=(X.shape[0],), fill_value=self.most_frequent_)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        # Ensure y is 1D
        y = np.ravel(y)

        # Generate predictions
        predictions = self.predict(X)

        # Calculate and return the accuracy
        return accuracy_score(y, predictions)

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load a dataset
iris = load_iris()
X, y = iris.data, iris.target

# Simplify to a binary classification problem
is_class_0_or_1 = y < 2
X_bin = X[is_class_0_or_1]
y_bin = y[is_class_0_or_1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# Initialize and fit the custom classifier
classifier = MostFrequentClassClassifier()
classifier.fit(X_train, y_train)

# Evaluate the classifier using the score method
score = classifier.score(X_test, y_test)
print(f"Accuracy of the MostFrequentClassClassifier: {score}")

"""### Transformers"""

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Generate some data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Use the transformer directly
X_transformed = StandardScaler().fit_transform(X)


LinearRegression().fit(X_transformed, y)

"""### Custom Transformer using Function Transformer"""

import numpy as np

def cube(x):

    return np.power(x,3)

from sklearn.preprocessing import FunctionTransformer

# Create the custom transformer
cube_transformer = FunctionTransformer(cube)

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Generate some data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Use the transformer directly
X_transformed = cube_transformer.transform(X)

LinearRegression().fit(X_transformed, y)

"""### Custom Transformer using BaseEstimator and TransformerMixin"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MedianIQRScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_ = None
        self.iqr_ = None

    def fit(self, X, y=None):
        # Calculate medians and interquartile range for each feature
        self.medians_ = np.median(X, axis=0)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = Q3 - Q1

        # Handle case where IQR is 0 to avoid division by zero during transform
        self.iqr_[self.iqr_ == 0] = 1
        return self

    def transform(self, X):
        # Check if fit has been called
        if self.medians_ is None or self.iqr_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        # Scale features using median and IQR learned during fit
        return (X - self.medians_) / self.iqr_

from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)

# Initialize the transformer
scaler = MedianIQRScaler()

# Fit the scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)

# Check the first few rows of the transformed data
print("Transformed data (first 5 rows):")
print(X_scaled[:5])

"""### Column Transformer"""

import pandas as pd

# Define the data with numeric labels for sentiment
data = {
    "Social Media Platform": ["Twitter", "Facebook", "Instagram", "Twitter", "Facebook",
                              "Instagram", "Twitter", "Facebook", "Instagram", "Twitter"],
    "Review": ["Love the new update!", "Too many ads now", "Great for sharing photos",
               "Newsfeed algorithm is biased", "Privacy concerns with latest update",
               "Amazing filters!", "Too much spam", "Easy to connect with friends",
               "Stories feature is fantastic", "Customer support lacking"],
    "age": [21, 19, np.nan, 17, 24, np.nan, 30, 19, 16, 31],
    "Sentiment": [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Numeric labels: 1 for Positive, 0 for Negative
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Define the column transformer
column_transformer = ColumnTransformer(
    transformers=[
        ('platform_ohe', OneHotEncoder(), ['Social Media Platform']),
        ('review_bow', CountVectorizer(), 'Review'),
        ('age_impute', SimpleImputer(),['age'])
    ],
    remainder='drop'  # Drop other columns not specified in transformers
)

pd.DataFrame(column_transformer.fit_transform(df).toarray(),columns=column_transformer.get_feature_names_out())

"""### Feature Union"""

import pandas as pd
import numpy as np

# Generating a random dataset with 10 rows and 4 columns
np.random.seed(42)  # For reproducibility
data = np.random.randn(10, 4)

# Creating a DataFrame and naming the columns
df = pd.DataFrame(data, columns=['f1', 'f2', 'f3', 'y'])

df

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA

# Define FeatureUnion
feature_union = FeatureUnion([
    ('scaler', StandardScaler()),  # Apply StandardScaler
    ('pca', PCA(n_components=2))   # Apply PCA, reduce to 2 components
])

X_transformed = feature_union.fit_transform(df.drop(columns=['y']))

pd.DataFrame(X_transformed, columns=feature_union.get_feature_names_out())

"""### Pipeline"""

import pandas as pd
import numpy as np

# Generating a random dataset with 10 rows and 4 columns
np.random.seed(42)  # For reproducibility
data = np.random.randn(10, 4)

# Creating a DataFrame and naming the columns
df = pd.DataFrame(data, columns=['f1', 'f2', 'f3', 'y'])

df

from sklearn.pipeline import Pipeline

# Define FeatureUnion
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply StandardScaler
    ('pca', PCA(n_components=2))
])

pd.DataFrame(pipeline.fit_transform(X), columns=pipeline.get_feature_names_out())

"""### Slightly Complex Example"""

import pandas as pd

# Define the data with numeric labels for sentiment
data = {
    "Social Media Platform": ["Twitter", "Facebook", "Instagram", "Twitter", "Facebook",
                              "Instagram", "Twitter", "Facebook", "Instagram", "Twitter"],
    "Review": ["Love the new update!", "Too many ads now", "Great for sharing photos",
               "Newsfeed algorithm is biased", "Privacy concerns with latest update",
               "Amazing filters!", "Too much spam", "Easy to connect with friends",
               "Stories feature is fantastic", "Customer support lacking"],
    "age": [21, 19, np.nan, 17, 24, np.nan, 30, 19, 16, 31],
    "Sentiment": [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Numeric labels: 1 for Positive, 0 for Negative
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)

def count_words(reviews):
    # Count the number of words in each review
    # Assuming reviews is a 1D array-like of text strings
    return np.array([len(review.split()) for review in reviews]).reshape(-1, 1)

from sklearn.preprocessing import FunctionTransformer

# Create the FunctionTransformer using the count_words function
word_count_transformer = FunctionTransformer(count_words)

feature_union = FeatureUnion([
    ('word_count', word_count_transformer),
    ('bag_of_words', CountVectorizer())
])

column_transformer = ColumnTransformer(
    transformers=[
        ('age_imputer', SimpleImputer(strategy='mean'), ['age']),
        ('platform_ohe', OneHotEncoder(), ['Social Media Platform']),
        ('review_processing', feature_union, 'Review')
    ],
    remainder='drop'  # Drop other columns not specified here
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest,chi2

final_pipeline = Pipeline(steps=[
    ('col_transformer', column_transformer),
    ('scaler', MaxAbsScaler()),
    ('selector', SelectKBest(score_func=chi2,k=10)),
    ('classifier', LogisticRegression())
])

final_pipeline.fit(df.drop(columns=['Sentiment']), df['Sentiment'])

