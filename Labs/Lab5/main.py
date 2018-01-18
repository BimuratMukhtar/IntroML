from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils

#-------------------------------------------------------------------------------
# Part 1 - Data Loading
#-------------------------------------------------------------------------------

iris = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

#-------------------------------------------------------------------------------
# Part 2 - Decision Tree Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 2.1 - Building and Training a Decision Tree Classifier
#-------------------------------------------------------------------------------

# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(X_train, y_train)

#-------------------------------------------------------------------------------
# Part 2.2 - Evaluating the Decision Tree Classifier
#-------------------------------------------------------------------------------

# train_predictions = dt.predict(X_train)
# train_accuracy = accuracy_score(y_train, train_predictions)
# print(train_accuracy)
#
# test_predictions = dt.predict(X_test)
# test_accuracy = accuracy_score(y_test, test_predictions)
# print(test_accuracy)

#-------------------------------------------------------------------------------
# Part 2.3 - Interpreting the Decision Tree Classifier
#-------------------------------------------------------------------------------

# print(utils.sort_features(iris.feature_names, dt.feature_importances_))
# utils.display_decision_tree(dt, iris.feature_names, iris.target_names)

#-------------------------------------------------------------------------------
# Part 3 - Random Forest Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Building and Training a Random Forest Classifier
#-------------------------------------------------------------------------------

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

#-------------------------------------------------------------------------------
# Part 3.2 - Evaluating the Random Forest Classifier
#-------------------------------------------------------------------------------

train_predictions = rfc.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(train_accuracy)

test_predictions = rfc.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(test_accuracy)

#-------------------------------------------------------------------------------
# Part 3.3 - Interpreting the Random Forest Classifier
#-------------------------------------------------------------------------------

for dt in rfc.estimators_:
    print(utils.sort_features(iris.feature_names, dt.feature_importances_))
    utils.display_decision_tree(dt, iris.feature_names, iris.target_names)