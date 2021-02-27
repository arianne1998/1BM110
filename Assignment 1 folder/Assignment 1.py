from pprint import pprint

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

campaign_df = pd.read_csv("campaign-data.csv")
binarize_columns = ["job", "mortgage", "marital"]
label_column = "y"
binary_depth = 10

# Every row which contains the String "unknown" will get the False which is inverted with the ~ operator
# Every value that is false will be removed from the dataset
filtered_campaign = campaign_df[~campaign_df.eq("unknown").any(1)]

# Split x (features) and y (labels) in separate dataframes
campaign_x = filtered_campaign.copy()
del campaign_x[label_column]
campaign_y = filtered_campaign.copy()[[label_column]]

# Ordinal encode education column
mapper = {"illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3, "high.school": 4, "university.degree": 5,
          "professional.course": 6}
campaign_x.education = campaign_x.education.replace(mapper)

# Binarize all categorical (string) columns
binarized_campaign_x = pd.get_dummies(campaign_x, columns=binarize_columns)
binarized_campaign_y = pd.get_dummies(campaign_y)

# Split dataframes into test and train with a ratio of 30%
train_x, test_x, train_y, test_y = train_test_split(binarized_campaign_x, binarized_campaign_y, test_size=.3,
                                                    random_state=0)

# Perform 10-fold cross validation
depth = []
for i in range(2, 11):
    clf = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=clf, X=train_x, y=train_y, cv=10)
    depth.append((i, scores.mean()))
pprint(f"10-fold cross validation result: {depth}")
best_depth = max(depth, key=lambda d: d[1])[0]
print(f"Optimal value of the parameter: {best_depth}")

# Create decision tree with best depth
clf = DecisionTreeClassifier(max_depth=best_depth)

# Train the decision tree model and predict with final test set
y_predicted = clf.fit(train_x, train_y).predict(test_x)
accuracy = accuracy_score(y_predicted, test_y)
print(f"Accuracy: {accuracy}")

# Calculate ROC AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(test_y.columns)):
    fpr[i], tpr[i], _ = roc_curve(test_y.iloc[:, i], y_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"ROC AUC {i} ({test_y.columns[i]}): {roc_auc[i]}")

# visualize diagram for tree
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf, class_names=train_y.columns, feature_names=train_x.columns, filled=True)
fig.savefig("decision_tree.png")
