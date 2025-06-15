import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Step 7️ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Step 8 Pairplot (Seaborn)
df_iris = pd.DataFrame(X, columns=feature_names)
df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
sns.pairplot(df_iris, hue='species', diag_kind='kde')
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()

# Step 9 Decision Boundary Plot (using first two features)
def plot_decision_boundary(X, y, clf, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    plt.scatter(
        X[:, 0], X[:, 1], c=y, s=40,
        cmap='Set1', edgecolor='k'
    )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.show()

import numpy as np
# Train classifier on first two features
X2 = X[:, :2]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)
sc2 = StandardScaler()
X2_train_scaled = sc2.fit_transform(X2_train)
X2_test_scaled = sc2.transform(X2_test)

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X2_train_scaled, y2_train)

plot_decision_boundary(
    np.vstack((X2_train_scaled, X2_test_scaled)),
    np.hstack((y2_train, y2_test)),
    clf2,
    title="Decision Boundary (Sepal Length vs Sepal Width)"
)
