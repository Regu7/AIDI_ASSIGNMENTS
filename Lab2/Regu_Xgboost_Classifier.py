# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# %%
from ucimlrepo import fetch_ucirepo

# %% [markdown]
# # Import Data


# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets


# %% [markdown]
# # Preprocessing

# %%
from sklearn.model_selection import train_test_split

# Initialize the encoder
encoder = LabelEncoder()

# Fit and transform the target variable
y = encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# # Model 1 Training

# %%
# Initialize the Logistic Regression model
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)


# %% [markdown]
# # Model 1 Evaluation

# %%
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
