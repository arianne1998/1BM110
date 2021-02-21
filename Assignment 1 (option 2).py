import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

campaign_df = pd.read_csv("campaign-data.csv")
binarize_columns = ["children", "job", "education", "mortgage", "marital", "y"]

# Every row which contains the String "unknown" will get the False which is inverted with the ~ operator
# Every value that is false will be removed from the dataset
filtered_campaign = campaign_df[~campaign_df.eq("unknown").any(1)]

# Binarize all categorical (string) columns
binarized_campaign = pd.get_dummies(filtered_campaign, columns=binarize_columns)

binarized_campaign_x = binarized_campaign.copy()
del binarized_campaign_x["y_yes"]
del binarized_campaign_x["y_no"]

binarized_campaign_y = binarized_campaign.copy()
binarized_campaign_y = binarized_campaign_y[["y_yes", "y_no"]]

train_x, test_x, train_y, test_y = train_test_split(binarized_campaign_x, binarized_campaign_y, test_size=.3, random_state=0)

# parameter minimal number of records equal to 10
classifier = DecisionTreeClassifier(max_depth=10)

y_score = classifier.fit(train_x, train_y).predict(test_x)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_y.iloc[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc[0])
