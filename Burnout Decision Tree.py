# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings('ignore')

# Create synthetic burnout-related dataset
data = {
    'SleepQuality': [5, 6, 4, 3, 7, 8, 2, 4, 6, 9]*10,
    'Workload': [45, 60, 55, 70, 40, 35, 80, 65, 50, 30]*10,
    'SocialInteraction': [3, 6, 5, 2, 8, 7, 1, 4, 6, 9]*10,
    'EmotionalCheckIn': [4, 5, 3, 2, 6, 8, 1, 4, 5, 9]*10,
    'BreathworkMinutes': [5, 10, 0, 2, 15, 20, 1, 3, 10, 25]*10,
    'ScreensBeforeBed': [3, 2, 5, 4, 1, 1, 6, 4, 2, 0]*10,
    'MovementPerDay': [20, 30, 10, 5, 40, 50, 0, 15, 25, 60]*10,
    'Burnout': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]*10
}

df = pd.DataFrame(data)

# Split into features and target
X = df.drop('Burnout', axis=1)
y = df['Burnout']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Visualize decision tree
plt.figure(figsize=(16, 10))
tree.plot_tree(clf, filled=True,
               feature_names=X.columns,
               class_names=['Not Burned Out', 'Burned Out'])
plt.title('Burnout Prediction Decision Tree')
plt.show()
