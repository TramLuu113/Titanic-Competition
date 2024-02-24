# (1) Data Understanding or Exploratory Data Analysis (EDA)
# (2) Data Preparation or Feature engineering
# (3) Modelling

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import datasets
training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# (1) Data Understanding or Exploratory Data Analysis (EDA):
# Rename the columns as lowercase
training.rename(
    columns={column: column.lower() for column in training.columns}, inplace=True)

test.rename(
    columns={column: column.lower() for column in test.columns}, inplace=True)

# Quick look at training data types and null counts
training.info()  # Age, cabin and embarked have missing values in training
test.info()  # Age, cabin and fare have missing values in training

# Then, use .describe() to understand the numeric data
training_describe = training.describe()
test_describe = test.describe()

# Brainstorm and suggest hypothesis by:
# Look closer at numeric and categorical values separately
training_num = training[['age', 'sibsp', 'parch', 'fare']]
training_cat = training[['survived', 'pclass', 'sex', 'ticket', 'cabin', 'embarked']]

# First, the numeric variables:
# Visualize the distributions for all numeric variables
for i in training_num.columns:
    plt.hist(training_num[i])
    plt.title(i)
    plt.show()

# Then, compare 'survive' with each of numeric variables using pivot table
pd.pivot_table(training, index='survived', values=['age', 'sibsp', 'parch', 'fare'])
# Hypothesis: the survival rate will be higher if that individual is younger / pay more money/
# have fewer siblings/ go with parents

# Second, the categorical variables:
# Visualize the distributions for all categorical variables
for i in training_cat.columns:
    sns.barplot(x=training_cat[i].value_counts().index, y=training_cat[i].value_counts().values)
    plt.title(f'{i}')
    plt.show()

# Then, compare 'survive' with each of categorical variables using pivot table:
print(pd.pivot_table(training, index='survived', columns='pclass', values='ticket', aggfunc='count'))
print()
# Individuals in first or third class may have better chance to survive
print(pd.pivot_table(training, index='survived', columns='sex', values='ticket', aggfunc='count'))
print()
# Individuals are female may have better chance to survive
print(pd.pivot_table(training, index='survived', columns='embarked', values='ticket', aggfunc ='count'))
# Individuals embarked at Southampton may have better chance to survive

# (2) Data Cleaning:
# (2.1) Training set:
# Before dropping the 'name', we think that the title might have some insights,
# e.g. individuals who have high position can have higher chance to survive
# So, split the 'name' column to have person's title
training['name_title'] = training.name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
training['name_title'].value_counts()

Title_Dictionary = {"Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty", "Don": "Royalty",
                    "Sir": "Royalty", "Dr": "Officer", "Rev": "Officer", "the Countess": "Royalty", "Mme": "Mrs",
                    "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
                    "Lady": "Royalty"}

training['name_title'] = training.name_title.map(Title_Dictionary)
training.head()
training.name_title.value_counts()

# Feature selection: drop unnecessary features
features_training = training.drop(['name', 'ticket', 'passengerid', 'cabin'], axis=1)
features_training.head()
features_training.info()

# Convert categorical variables into numerical
features_training.sex = features_training.sex.map({'female': 0, 'male': 1})
features_training.embarked = features_training.embarked.map({'S': 0, 'C': 1, 'Q': 2, 'nan': 'NaN'})
features_training.name_title = features_training.name_title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
                                                                 'Officer': 4, 'Royalty': 5})
features_training.head()
features_training.isnull().sum()

# Fill missing value 'age' and 'embarked'
features_training['age'].fillna(features_training['age'].mean(), inplace=True)
features_training['embarked'].fillna(features_training['embarked'].mean(), inplace=True)

# Scaling
features_training.age = (features_training.age-min(features_training.age))/(max(features_training.age)
                                                                            - min(features_training.age))
features_training.fare = (features_training.fare-min(features_training.fare))/(max(features_training.fare)
                                                                               - min(features_training.fare))
features_training.describe()
features_training.isnull().sum()

# (2.2) Test set:
# Split the 'name' column to have person's title
test['name_title'] = test.name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
test['name_title'].value_counts()
test['name_title'] = test.name_title.map(Title_Dictionary)
test.head()

# Feature selection: drop unnecessary features
features_test = test.drop(['name', 'ticket', 'passengerid', 'cabin'], axis=1)
features_test.head()
features_test.isnull().sum()

# Convert categorical variables into numerical
features_test.sex = features_test.sex.map({'female': 0, 'male': 1})
features_test.embarked = features_test.embarked.map({'S': 0, 'C': 1, 'Q': 2})
features_test.name_title = features_test.name_title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Officer': 4,
                                                         'Royalty': 5})
features_test.head()
features_test.isnull().sum()

# Impute the nulls for 'age' and 'fare' column
features_test['age'].fillna(features_test['age'].mean(), inplace=True)
features_test['fare'].fillna(features_test['fare'].mean(), inplace=True)

# Null value in the 'name_title' column
var = features_test[features_test['name_title'].isnull()]
features_test = features_test.fillna(2)
features_test.isnull().sum()

# Scaling
features_test.age = (features_test.age-min(features_test.age))/(max(features_test.age)-min(features_test.age))
features_test.fare = (features_test.fare-min(features_test.fare))/(max(features_test.fare)-min(features_test.fare))
features_test.describe()

# (3) Modelling
# Split the data into training and validation sets
X = features_training.drop(['survived'], axis=1)
y = features_training['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LogisticRegression
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Validation Accuracy:', accuracy)

# Make predictions on the test data
submission_preds = clf.predict(features_test)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    'PassengerId': test['passengerid'],
    'Survived': submission_preds
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission_7.csv', index=False)

# Random Forest
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)

randon_forest_preds = random_forest.predict(features_test)
submission_rf = pd.DataFrame({
    'PassengerId': test['passengerid'],
    'Survived': randon_forest_preds
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission_rf.csv', index=False)
