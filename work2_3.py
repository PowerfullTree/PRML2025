import numpy as np
import matplotlib.pyplot as plt

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate the dataset
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)

# Step 2: Split the dataset for training and testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 3: Define the classifiers
# Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)

# AdaBoost with Decision Tree as base estimator
ada_boost_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42),n_estimators=70, random_state=42)

# Support Vector Machine with different kernel functions
svm_linear = SVC(kernel='linear', random_state=42)
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_sigmoid = SVC(kernel='sigmoid')

# Step 4: Train and evaluate models

# Train and evaluate Decision Tree
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_val)
accuracy_dt = accuracy_score(y_val, y_pred_dt)

# Train and evaluate AdaBoost
ada_boost_classifier.fit(X_train, y_train)
y_pred_ada = ada_boost_classifier.predict(X_val)
accuracy_ada = accuracy_score(y_val, y_pred_ada)

# Train and evaluate SVM with linear kernel
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_val)
accuracy_svm_linear = accuracy_score(y_val, y_pred_svm_linear)

# Train and evaluate SVM with RBF kernel
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_val)
accuracy_svm_rbf = accuracy_score(y_val, y_pred_svm_rbf)

# Train and evaluate SVM with polynomial kernel
svm_poly.fit(X_train, y_train)
y_pred_svm_poly = svm_poly.predict(X_val)
accuracy_svm_poly = accuracy_score(y_val, y_pred_svm_poly)

# Train and evaluate SVM with polynomial kernel
svm_sigmoid.fit(X_train, y_train)
y_pred_svm_sigmoid = svm_sigmoid.predict(X_val)
accuracy_svm_sigmoid = accuracy_score(y_val, y_pred_svm_sigmoid)

# Step 5: Compare the results
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"AdaBoost + Decision Tree Accuracy: {accuracy_ada:.4f}")
print(f"SVM (Linear Kernel) Accuracy: {accuracy_svm_linear:.4f}")
print(f"SVM (RBF Kernel) Accuracy: {accuracy_svm_rbf:.4f}")
print(f"SVM (Polynomial Kernel) Accuracy: {accuracy_svm_poly:.4f}")
print(f"SVM (Sigmoid Kernel) Accuracy: {accuracy_svm_sigmoid:.4f}")

# Step 6: Test on the test set
y_pred_test_dt = dt_classifier.predict(X_test)
y_pred_test_ada = ada_boost_classifier.predict(X_test)
y_pred_test_svm_linear = svm_linear.predict(X_test)
y_pred_test_svm_rbf = svm_rbf.predict(X_test)
y_pred_test_svm_poly = svm_poly.predict(X_test)
y_pred_test_svm_sigmoid = svm_sigmoid.predict(X_test)

accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
accuracy_test_ada = accuracy_score(y_test, y_pred_test_ada)
accuracy_test_svm_linear = accuracy_score(y_test, y_pred_test_svm_linear)
accuracy_test_svm_rbf = accuracy_score(y_test, y_pred_test_svm_rbf)
accuracy_test_svm_poly = accuracy_score(y_test, y_pred_test_svm_poly)
accuracy_test_svm_sigmoid = accuracy_score(y_test, y_pred_test_svm_sigmoid)

# Display test accuracy
print("\nTest Set Accuracy:")
print(f"Decision Tree Test Accuracy: {accuracy_test_dt:.4f}")
print(f"AdaBoost + Decision Tree Test Accuracy: {accuracy_test_ada:.4f}")
print(f"SVM (Linear Kernel) Test Accuracy: {accuracy_test_svm_linear:.4f}")
print(f"SVM (RBF Kernel) Test Accuracy: {accuracy_test_svm_rbf:.4f}")
print(f"SVM (Polynomial Kernel) Test Accuracy: {accuracy_test_svm_poly:.4f}")
print(f"SVM (Sigmoid Kernel) Test Accuracy: {accuracy_test_svm_sigmoid:.4f}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_predictions(X, y_true, y_pred, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    correct_indices = np.where(y_true == y_pred)[0]
    ax.scatter(X[correct_indices, 0], X[correct_indices, 1], X[correct_indices, 2],
               c=y_true[correct_indices], cmap='viridis', marker='o')

    # Plot wrong predicted spots
    incorrect_indices = np.where(y_true != y_pred)[0]
    ax.scatter(X[incorrect_indices, 0], X[incorrect_indices, 1], X[incorrect_indices, 2],
               c='red', cmap='viridis', marker='x', s=100, label='Incorrect')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_predictions(X_test,y_test,y_pred_test_dt,'Decision Tree')
plot_predictions(X_test,y_test,y_pred_test_ada,'Adaboost Decision Tree')
plot_predictions(X_test,y_test,y_pred_test_svm_linear,'SVM linear')
plot_predictions(X_test,y_test,y_pred_test_svm_poly,'SVM poly')
plot_predictions(X_test,y_test,y_pred_test_svm_rbf,'SVM RBF')
plot_predictions(X_test,y_test,y_pred_test_svm_sigmoid,'SVM Sigmoid')