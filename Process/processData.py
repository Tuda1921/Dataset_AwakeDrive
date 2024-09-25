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
import scipy as sp
from scipy.interpolate import interp1d

import os


def filter_data(data):
    # Bandpass filter
    band = [0.5 / (0.5 * 512), 40 / (0.5 * 512)]
    b, a = sp.signal.butter(4, band, btype='band', analog=False, output='ba')
    data = sp.signal.lfilter(b, a, data)

    # plt.hist(data, bins=10, edgecolor='black')
    # filter for EMG by interpolated
    filtered_data = data[(np.abs(data) <= 512)]
    x = np.arange(len(filtered_data))
    interpolated_data = interp1d(x, filtered_data)(np.linspace(0, len(filtered_data) - 1, len(data)))
    return interpolated_data


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
    return np.sqrt(np.mean(signal ** 2))


def energy(signal):
    """Calculate the energy of a signal."""
    return np.sum(signal ** 2)


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
    indices1 = np.where((f1 >= 13) & (f1 <= 40))[0]
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


def check(data):
    has_greater_than_512 = np.any(data > 512)
    return has_greater_than_512

def slide(subject_data):
    # print(subject_data.shape)
    # Khởi tạo biến
    x = 0
    y = []
    fs = 512  # Tần số lấy mẫu
    k = 15 * fs  # Độ dài của 1 cửa sổ trượt (sliding window)

    features = []
    global NAMES
    NAMES = None
    # Đọc và xử lý dữ liệu
    while x < (len(subject_data)):
        x += 1  # Tăng giá trị x
        if x >= k:  # Khi đã thu thập đủ dữ liệu cho một cửa sổ trượt
            if x % (1 * 512) == 0:
                sliding_window_start = x - k  # Vị trí bắt đầu của cửa sổ trượt
                sliding_window_end = x  # Vị trí kết thúc của cửa sổ trượt
                sliding_window = np.array(subject_data[
                                          sliding_window_start:sliding_window_end])  # Tạo mảng sliding_window tương ứng với cửa sổ trượt trong y
                # sliding_window = filter(sliding_window)  # Áp dụng bộ lọc (nếu cần)
                # feature_test.append(FeatureExtract(
                #     sliding_window))  # Trích xuất đặc trưng và định hình lại để đưa vào mô hình
                if check(sliding_window) == False:
                    feature = FeatureExtract(sliding_window)
                    features.append(list(feature.values()))

                    if NAMES is None:
                        NAMES = list(feature.keys())

    return features

def slide2(subject_data):
    # print(subject_data.shape)
    # Khởi tạo biến
    x = 0
    y = []
    fs = 512  # Tần số lấy mẫu
    k = 15 * fs  # Độ dài của 1 cửa sổ trượt (sliding window)

    features = []
    global NAMES
    NAMES = None
    # Đọc và xử lý dữ liệu
    while x < (len(subject_data)):
        x += 1  # Tăng giá trị x
        if x >= k:  # Khi đã thu thập đủ dữ liệu cho một cửa sổ trượt
            if x % (15 * 512) == 0:
                sliding_window_start = x - k  # Vị trí bắt đầu của cửa sổ trượt
                sliding_window_end = x  # Vị trí kết thúc của cửa sổ trượt
                sliding_window = np.array(subject_data[
                                          sliding_window_start:sliding_window_end])  # Tạo mảng sliding_window tương ứng với cửa sổ trượt trong y
                # sliding_window = filter(sliding_window)  # Áp dụng bộ lọc (nếu cần)
                # feature_test.append(FeatureExtract(
                #     sliding_window))  # Trích xuất đặc trưng và định hình lại để đưa vào mô hình
                if check(sliding_window) == False:
                    feature = FeatureExtract(sliding_window)
                    features.append(list(feature.values()))

                    if NAMES is None:
                        NAMES = list(feature.keys())

    return features


def processData(folder_path):
    print(os.listdir(folder_path))
    try:
        os.listdir(folder_path).remove(".DS_Store")
    except:
        pass

    for index, dir in enumerate(os.listdir(folder_path)):
        subject_path = os.path.join(folder_path, dir)
        print(subject_path)
        if index == 0:
            task1_data, task2_data, task3_data, task4_data, task5_data = loadSubjectData(subject_path)
            X1 = slide2(task1_data)
            X2 = slide(task2_data)
            X3 = slide(task3_data)
            X4 = slide(task4_data)
            X5 = slide(task5_data)

        else:
            task1_data, task2_data, task3_data, task4_data, task5_data = loadSubjectData(subject_path)
            X1 += slide2(task1_data)
            X2 += slide(task2_data)
            X3 += slide(task3_data)
            X4 += slide(task4_data)
            X5 += slide(task5_data)

            # X2.append(slide(task2_data))
            # X3.append(slide(task3_data))
            # X4.append(slide(task4_data))
            # X5.append(slide(task5_data))

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)

    return X1, X2, X3, X4, X5

    # else:
    #     data = loadSubjectData(subject_path)
    #     # task1_data = np.concatenate((task1_data, loadSubjectData(subject_path)[0]))
    #     task2_data = np.concatenate((task2_data, data[1]))
    #     # print(task2_data)
    #     task3_data = np.concatenate((task3_data, data[2]))
    #     task4_data = np.concatenate((task4_data, data[3]))
    #     task5_data = np.concatenate((task5_data, data[4]))
    # return task1_data, task2_data, task3_data, task4_data, task5_data


def loadSubjectData(subject_path):
    task1_data = []
    task2_data = []
    task3_data = []
    task4_data = []
    task5_data = []

    for filename in os.listdir(subject_path):
        # Kiểm tra nếu là file .txt
        if filename.endswith(".txt"):
            # Đọc dữ liệu từ file
            file_data = np.loadtxt(os.path.join(subject_path, filename))

        match filename[:-4]:
            case "task1":
                task1_data.append(file_data)
            case "task2":
                task2_data.append(file_data)
            case "task3":
                task3_data.append(file_data)
            case "task4":
                task4_data.append(file_data)
            case "task5":
                task5_data.append(file_data)
            case _:
                print("file name is not correct!")
    try:
        task1_data = np.concatenate(task1_data, axis=0)
        print('task1')
    except:
        pass
    try:
        task2_data = np.concatenate(task2_data, axis=0)
    except:
        print(subject_path, 2)
        pass
    try:
        task3_data = np.concatenate(task3_data, axis=0)
    except:
        print(subject_path, 3)
        pass
    try:
        task4_data = np.concatenate(task4_data, axis=0)
    except:
        print(subject_path, 4)
        pass
    try:
        task5_data = np.concatenate(task5_data, axis=0)
    except:
        print(subject_path, 5)
        pass

    return task1_data, task2_data, task3_data, task4_data, task5_data


def saveToCSV(data_path, csv_path):
    X1, X2, X3, X4, X5 = processData(data_path)

    X = pd.DataFrame(X1)
    # X = pd.DataFrame(X2)
    X = pd.concat([X, pd.DataFrame(X2)], axis=0)
    X = pd.concat([X, pd.DataFrame(X3)], axis=0)
    X = pd.concat([X, pd.DataFrame(X4)], axis=0)
    X = pd.concat([X, pd.DataFrame(X5)], axis=0)

    y = pd.concat([
        pd.Series([0] * len(X1)),
        pd.Series([1] * len(X2)),
        pd.Series([2] * len(X3)),
        pd.Series([3] * len(X4)),
        pd.Series([4] * len(X5))])

    csv_file = pd.concat([X, y], axis=1)
    csv_file.columns = NAMES + ["label"]
    # csv_file.rename(columns = {0 : "label"}, inplace = True)
    print(csv_file)

    # dir_name = os.path.dirname(csv_path)
    # Create the directory if it does not exist
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    csv_file.to_csv(f"{csv_path}/Feature_full_label.csv", index=False)


if __name__ == "__main__":
    data_path = "../AD_UI/Data"
    csv_path = "CSV"
    saveToCSV(data_path, csv_path)

    # processData(data_path)
