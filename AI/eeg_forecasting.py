import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import pickle
import matplotlib.pyplot as plt

import scipy as sp
from scipy.interpolate import interp1d
from tqdm import tqdm
import scipy.stats
import numpy as np
import scipy.stats as stats
import pywt
from statsmodels.tsa.ar_model import AutoReg
import scipy.signal
from scipy.signal import find_peaks

# 1. Tải tín hiệu EEG từ tệp task2.txt
signal = np.loadtxt('task1.txt')

# Xác định các thông số
sample_rate = 512  # Hz
window_size = 15 * sample_rate  # 15 giây
overlap_size = 14 * sample_rate  # 14 giây chồng chéo

# Danh sách để lưu các cửa sổ
windows = []

with open('model.pkl', 'rb') as f:
    model1 = pickle.load(f)
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

# Tạo các cửa sổ
prob_check = model1.predict_proba(np.array(slide(filter_data(signal))))
out = [0, 25, 50, 75, 100]
prob_check = np.dot(prob_check, out)
print(prob_check)
print(prob_check.shape)

# Chuyển đổi danh sách thành mảng NumPy
windows = np.array(prob_check)

# 4. Tạo dữ liệu cho Model 2 (seq2seq)
def create_sequences(data, n_steps_input, n_steps_output):
    X, y = [], []
    for i in range(len(data)):
        X.append(i)
        y.append(data[i])
    return np.array(X), np.array(y)

n_steps_input = 1  # Kích thước cửa sổ đầu vào (15 giây)
n_steps_output = 1  # Kích thước đầu ra (dự đoán trong 10 giây tới)

# Tạo dữ liệu cho mô hình seq2seq
X, y = create_sequences(windows, n_steps_input, n_steps_output)
print(X.shape)
print(y.shape)
# Chia dữ liệu thành tập huấn luyện và kiểm tra
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape dữ liệu đầu vào để phù hợp với định dạng LSTM
X_train = X_train.reshape((X_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, 1))

# 5. Định nghĩa mô hình seq2seq (Model 2)
encoder_inputs = layers.Input(shape=(n_steps_input, 1))
encoder_lstm = layers.LSTM(128, activation='tanh', return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# Decoder
decoder_inputs = layers.RepeatVector(n_steps_output)(state_h)  # Lặp lại trạng thái 10 lần
decoder_lstm = layers.LSTM(128, activation='tanh', return_sequences=True)  # Trả về các chuỗi
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
decoder_dense = layers.TimeDistributed(layers.Dense(1))
decoder_outputs = decoder_dense(decoder_outputs)

# Biên dịch mô hình
model2 = models.Model(encoder_inputs, decoder_outputs)
model2.compile(optimizer='adam', loss='mse')

# 6. Huấn luyện mô hình seq2seq (Model 2)
history = model2.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss = model2.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Dự đoán
y_pred = model2.predict(X_test)
print(len(y_pred))
# Vẽ đồ thị
plt.figure(figsize=(10, 6))
X_train = X_train.reshape(-1)
X_test = X_test.reshape(-1)
y_pred = y_pred.reshape(-1)
# Vẽ chuỗi đầu vào
plt.plot(X_train[:], y_train, label='Input Sequence (Past)')

# Vẽ chuỗi thực tế
plt.plot(X_test[:], y_test, label='True Future Sequence')

# Vẽ chuỗi dự đoán
plt.plot(X_test[:], y_pred, label='Predicted Future Sequence')

plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Alertness Level')
plt.title('Seq2Seq Prediction of Alertness')
plt.show()

# Vẽ chuỗi thực tế
plt.plot(X_test[:], y_test, label='True Future Sequence')

# Vẽ chuỗi dự đoán
plt.plot(X_test[:], y_pred, label='Predicted Future Sequence')
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Alertness Level')
plt.title('Seq2Seq Prediction of Alertness')
plt.show()