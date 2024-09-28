import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold  # Import KFold
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from xgboost import XGBClassifier  # Import XGBoost

# Load the dataset
X = pd.read_csv('..\Process\CSV\Feature_full_label.csv')
print('Load_csv: Done')
y = X['label']
label_counts = y.value_counts()
print("Số lượng từng loại nhãn:")
print(label_counts)
X = X.drop('label', axis=1)
X = X.values

# Initialize the XGBoost model
model = XGBClassifier(n_estimators=100, random_state=42, verbosity=1)

# Initialize KFold with 5 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store results
train_scores = []
test_scores = []

# Perform KFold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")

    # Split the data into training and testing sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model on the current fold
    model.fit(X_train, y_train.values)
    print('Model.fit: Done')

    # Evaluate the model on the training and testing sets for this fold
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Save the scores
    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"Train score for fold {fold + 1}: {train_score}")
    print(f"Test score for fold {fold + 1}: {test_score}\n")

# Calculate the mean and standard deviation of train and test scores
print(f"Mean train score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
print(f"Mean test score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

# Save the model to a file (optional: after the final fold, or from the best fold)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict labels on the last test fold
y_pred = np.array(model.predict(X_test))

# Create confusion matrix
label_mapping = {
    "Task1": 0,
    "Task2": 1,
    "Task3": 2,
    "Task4": 3,
    "Task5": 4}
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification report
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(5) + 0.5, label_mapping.keys())
plt.yticks(np.arange(5) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(f"Classification Report for {type(model).__name__}:\n----------------------\n", clr)

# Task 1 test data processing
task1_test = np.loadtxt('../AD_UI/Task1/Task1_dat/task1.txt')
from Process.processData import slide, filter_data

task1_test = pd.DataFrame(slide(filter_data(task1_test)))

# Predict probability on task1 data
prob_check = model.predict_proba(np.array(task1_test))
out = [0, 25, 50, 75, 100]
prob_check = np.dot(prob_check, out)
from scipy.ndimage import gaussian_filter

prob_check = gaussian_filter(prob_check, sigma=100)

# Plot probability check results
plt.plot(prob_check)

# Add vertical lines for iterations divisible by 5400
for iter in range(0, len(prob_check), 5400):
    plt.axvline(x=iter, color='r', linestyle='--')

# Show plot
plt.show()
