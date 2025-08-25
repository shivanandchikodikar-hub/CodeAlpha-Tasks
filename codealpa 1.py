#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.read_csv("Iris.csv")  

X = iris_df.drop(columns=['Id', 'Species'])
y = iris_df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance in Iris Classification")
plt.show()


# In[ ]:




