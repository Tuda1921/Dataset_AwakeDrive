import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
# print(type(X2), X2)
# X = pd.DataFrame(X1)
X = pd.read_csv('..\Process\CSV\Feature_full_label.csv')
print('Load_csv: Done')
y = X['label']
X = X.drop('label', axis=1)

# In ra dữ liệu đầu vào để kiểm tra
# print(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Split Data: Done')
# Khởi tạo mô hình SVM với kernel tuyến tính
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# Huấn luyện mô hình với tập dữ liệu huấn luyện
model.fit(X_train.values, y_train.values)
print('Model.fit: Done')
# Tính điểm (score) của mô hình trên tập huấn luyện và tập kiểm tra
train_score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
print("Train score: ", train_score)
print("Test score: ", score)

probabilities = model.predict_proba(X)
out = [0, 25, 50, 75, 100]
probabilities = np.dot(probabilities, out)
# print(np.round(probabilities, 0))
plt.plot(probabilities)
plt.show()
# Dự đoán nhãn trên tập dữ liệu kiểm tra
y_pred = np.array(model.predict(X_test))
# Tạo ma trận nhầm lẫn để đánh giá kết quả mô hình
label_mapping = {
    "Task1": 0,
    "Task2": 1,
    "Task3": 2,
    "Task4": 3,
    "Task5": 4}
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Tạo báo cáo phân loại chi tiết
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(5) + 0.5, label_mapping.keys())
plt.yticks(np.arange(5) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()  # Hiển thị biểu đồ

# In báo cáo phân loại
print(f"Classification Report for {type(model).__name__}:\n----------------------\n", clr)

# temp = np.array(slide(('Data/Thanh3/Task5.txt')))
# temp = pd.DataFrame(temp)
# prob_temp = model.predict_proba(temp)
# prob_temp = np.dot(prob_temp, out)
# # print(np.round(prob_temp,2))
# plt.plot(prob_temp)
# plt.show()
# KFold Cross-Validation (mã mẫu, có thể kích hoạt nếu cần)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Định nghĩa số lượng fold
k = 10

# Khởi tạo đối tượng KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Khởi tạo danh sách để lưu điểm accuracy
accuracy_list = []

# Vòng lặp qua từng fold
for train_index, test_index in kf.split(X):
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Khởi tạo mô hình
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=1)

    # Huấn luyện mô hình trên tập dữ liệu huấn luyện
    model.fit(X_train, y_train)

    # Dự đoán trên tập dữ liệu kiểm tra
    y_pred = model.predict(X_test)

    # Tính điểm accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Thêm điểm accuracy vào danh sách
    accuracy_list.append(accuracy)
    print(accuracy)

# Tính điểm accuracy trung bình
avg_accuracy = np.mean(accuracy_list)
print("Average Accuracy Score:", avg_accuracy)
