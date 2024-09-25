from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import scipy.stats
import scipy.signal
from scipy.signal import find_peaks

def shannon_entropy(signal):
    """Calculate the Shannon entropy of a signal."""
    probability_distribution, _ = np.histogram(signal, bins='fd', density=True)
    probability_distribution = probability_distribution[probability_distribution > 0]
    entropy = -np.sum(probability_distribution * np.log2(probability_distribution))
    return entropy

def higuchi_fd(signal, kmax=10):
    """Calculate the Higuchi Fractal Dimension of a signal."""
    N = len(signal)
    Lk = np.zeros((kmax,))

    for k in range(1, kmax + 1):
        Lmk = np.zeros((k,))
        for m in range(k):
            Lm = 0
            n_max = int(np.floor((N - m - 1) / k))
            for j in range(1, n_max):
                Lm += np.abs(signal[m + j * k] - signal[m + (j - 1) * k])
            Lmk[m] = (Lm * (N - 1) / (k * n_max)) / k

        Lk[k - 1] = np.mean(Lmk)

    Lk = np.log(Lk)
    ln_k = np.log(np.arange(1, kmax + 1))
    higuchi, _ = np.polyfit(ln_k, Lk, 1)

    return higuchi

def zero_crossing_rate(signal):
    """Calculate the zero-crossing rate of a signal."""
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    return len(zero_crossings) / len(signal)

def root_mean_square(signal):
    """Calculate the root mean square (RMS) of a signal."""
    return np.sqrt(np.mean(signal**2))

def energy(signal):
    """Calculate the energy of a signal."""
    return np.sum(signal**2)

def envelope(signal):
    """Calculate the envelope of a signal using Hilbert transform."""
    analytic_signal = scipy.signal.hilbert(signal)
    return np.abs(analytic_signal)

def autocorrelation(signal):
    """Calculate the autocorrelation of a signal."""
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:]

def peak_analysis(signal):
    """Calculate the peak analysis features of a signal."""
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    peak_to_peak_val = np.ptp(signal)
    return peak_count, peak_to_peak_val

def spectral_entropy(signal, sf):
    """Calculate the spectral entropy of a signal."""
    _, psd = scipy.signal.welch(signal, sf)
    psd_norm = psd / psd.sum()
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return spectral_entropy


def hurst_exponent(signal):
    """Calculate the Hurst exponent of a signal."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(signal[lag:], signal[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def approximate_entropy(signal):
    """Calculate the approximate entropy of a signal."""
    N = len(signal)
    m = 2
    r = 0.2 * np.std(signal)
    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)
    return _phi(m) - _phi(m + 1)

def FeatureExtract(y, sf=512):
    L = len(y)  # Length of signal

    Y = np.fft.fft(y)  # Perform FFT
    Y[0] = 0  # Set DC component to zero
    P2 = np.abs(Y / L)  # Two-sided spectrum
    P1 = P2[:L // 2 + 1]  # One-sided spectrum
    P1[1:-1] = 2 * P1[1:-1]  # Adjust FFT spectrum

    # Frequency ranges
    f1 = np.arange(len(P1)) * sf / len(P1)
    indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]
    delta = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]
    theta = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]
    alpha = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 13) & (f1 <= 30))[0]
    beta = np.sum(P1[indices1])

    # Feature ratios
    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (delta + theta) / (alpha + beta)

    # Basic statistical features
    mean_val = np.mean(y)
    variance_val = np.var(y)
    min_val = np.min(y)
    max_val = np.max(y)
    skewness_val = scipy.stats.skew(y)
    kurtosis_val = scipy.stats.kurtosis(y)

    # Hjorth parameters
    activity, mobility, complexity = hjorth_parameters(y)

    # Non-linear features
    entropy_val = shannon_entropy(y)
    # fractal_dim_val = higuchi_fd(y)

    # Time-domain features
    # zero_cross_rate = zero_crossing_rate(y)
    # rms_val = root_mean_square(y)
    # energy_val = energy(y)
    # envelope_val = np.mean(envelope(y))
    # autocorr_val = np.mean(autocorrelation(y))
    #
    # # Peak analysis
    # peak_count, peak_to_peak_val = peak_analysis(y)
    #
    # # Spectral features
    # spectral_entropy_val = spectral_entropy(y, sf)
    #
    # # Additional non-linear features
    # hurst_val = hurst_exponent(y)
    # approx_entropy_val = approximate_entropy(y)

    # Create feature dictionary
    features_dict = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "abr": abr,
        "tbr": tbr,
        "dbr": dbr,
        "tar": tar,
        "dar": dar,
        "dtabr": dtabr,
        "mean": mean_val,
        "variance": variance_val,
        "min": min_val,
        "max": max_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "entropy": entropy_val,
        # "fractal_dimension": fractal_dim_val,
        # "zero_crossing_rate": zero_cross_rate,
        # "rms": rms_val,
        # "energy": energy_val,
        # "envelope": envelope_val,
        # "autocorrelation": autocorr_val,
        # "peak_count": peak_count,
        # "peak_to_peak": peak_to_peak_val,
        # "spectral_entropy": spectral_entropy_val,
        # "hurst_exponent": hurst_val,
        # "approximate_entropy": approx_entropy_val,
    }

    return features_dict

def hjorth_parameters(y):
    """Calculate the Hjorth parameters of a signal."""
    first_deriv = np.diff(y)
    second_deriv = np.diff(y, 2)

    activity = np.var(y)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility

    return activity, mobility, complexity



def slide(path):
    # Khởi tạo biến
    x = 0
    y = []
    fs = 512  # Tần số lấy mẫu
    k = 15 * fs  # Độ dài của 1 cửa sổ trượt (sliding window)

    # Đường dẫn đến file dữ liệu cần xét
    file = open(path, "r")  # Mở file dữ liệu
    feature_test = []
    # Đọc và xử lý dữ liệu
    while x < (180*fs):  # Lặp qua các giá trị dữ liệu
        if x % fs == 0:
            print(x // fs)  # In ra số giây hiện tại
        data = file.readline()  # Đọc một dòng dữ liệu từ file
        if data == '':
            break
        if data == '\n':
            continue

        y = np.append(y, int(data))  # Thêm dữ liệu vào mảng y
        x += 1  # Tăng giá trị x
        if x >= k:  # Khi đã thu thập đủ dữ liệu cho một cửa sổ trượt
            if x % (1 * 512) == 0:
                sliding_window_start = x - k  # Vị trí bắt đầu của cửa sổ trượt
                sliding_window_end = x  # Vị trí kết thúc của cửa sổ trượt
                sliding_window = np.array(y[
                                          sliding_window_start:sliding_window_end])  # Tạo mảng sliding_window tương ứng với cửa sổ trượt trong y
                # sliding_window = filter(sliding_window)  # Áp dụng bộ lọc (nếu cần)
                feature_test.append(list(FeatureExtract(
                    sliding_window).values()))  # Trích xuất đặc trưng và định hình lại để đưa vào mô hình
    return feature_test


# print(slide((r'Data/PY_VAR0/task2.txt')))
#
# X1 = np.array(slide(('Data/Dat2/task1.txt')))
X2 = np.array(slide(('../AD_UI/Data/Chau1/task2.txt')))
X3 = np.array(slide(('../AD_UI/Data/Chau1/task3.txt')))
X4 = np.array(slide(('../AD_UI/Data/Chau1/task4.txt')))
X5 = np.array(slide(('../AD_UI/Data/Chau1/task5.txt')))
print(len(X2), len(X3), len(X5))
# print(type(X2), X2)
# X = pd.DataFrame(X1)
X = pd.DataFrame(X2)
# X = pd.concat([X, pd.DataFrame(X2)], axis=0)
X = pd.concat([X, pd.DataFrame(X3)], axis=0)
X = pd.concat([X, pd.DataFrame(X4)], axis=0)
X = pd.concat([X, pd.DataFrame(X5)], axis=0)
y = pd.concat([
               # pd.Series([0] * len(X1)),
               pd.Series([1] * len(X2)),
               pd.Series([2] * len(X3)),
               pd.Series([3] * len(X4)),
               pd.Series([4] * len(X5))])

# In ra dữ liệu đầu vào để kiểm tra
# print(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình SVM với kernel tuyến tính
model = svm.SVC(kernel="linear", probability=True, random_state=42)
# Huấn luyện mô hình với tập dữ liệu huấn luyện
model.fit(X_train.values, y_train.values)


# Tính điểm (score) của mô hình trên tập huấn luyện và tập kiểm tra
train_score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
print("Train score: ", train_score)
print("Test score: ", score)

probabilities = model.predict_proba(X)
out = [0, 33, 66, 100]
probabilities = np.dot(probabilities, out)
# print(np.round(probabilities, 0))
plt.plot(probabilities)
plt.show()
# Dự đoán nhãn trên tập dữ liệu kiểm tra
y_pred = np.array(model.predict(X_test))
# Tạo ma trận nhầm lẫn để đánh giá kết quả mô hình
label_mapping = {
    # "Task1": 0,
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
plt.xticks(np.arange(4) + 0.5, label_mapping.keys())
plt.yticks(np.arange(4) + 0.5, label_mapping.keys())
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
kf = KFold(n_splits=k)

# Khởi tạo danh sách để lưu điểm accuracy
accuracy_list = []

# Vòng lặp qua từng fold
for train_index, test_index in kf.split(X):
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Khởi tạo mô hình
    model = svm.SVC(kernel="linear")

    # Huấn luyện mô hình trên tập dữ liệu huấn luyện
    model.fit(X_train, y_train)

    # Dự đoán trên tập dữ liệu kiểm tra
    y_pred = model.predict(X_test)

    # Tính điểm accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Thêm điểm accuracy vào danh sách
    accuracy_list.append(accuracy)

# Tính điểm accuracy trung bình
avg_accuracy = np.mean(accuracy_list)
print("Average Accuracy Score:", avg_accuracy)
