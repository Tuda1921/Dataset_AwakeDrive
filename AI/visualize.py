import numpy as np  # Import thư viện numpy để làm việc với mảng và ma trận
import matplotlib.pyplot as plt  # Import thư viện matplotlib để vẽ đồ thị
import scipy as sp  # Import thư viện scipy để sử dụng các chức năng xử lý tín hiệu
from scipy.interpolate import interp1d
data = []  # Khởi tạo một danh sách rỗng để lưu trữ dữ liệu

# Mở tệp tin "Tran_tinhtao.txt" để đọc
with open(r"C:\Users\nguye\OneDrive - Hanoi University of Science and Technology\Documents\Research\Awake Drive\Dataset_AwakeDrive\AD_UI\Data\Quyen3\task2.txt", "r") as file:
    # Đọc từng dòng trong tệp tin và thêm vào danh sách data sau khi chuyển đổi thành số nguyên
    for value in file:
        data.append(int(value))

# print(data)  # In danh sách dữ liệu ra màn hình

# plt.plot(data)  # Vẽ đồ thị dữ liệu
# plt.ylim(-255, 255)  # Đặt giới hạn trục y trong khoảng -255 đến 255
# plt.show()  # Hiển thị đồ thị

fs = 512  # Tần số lấy mẫu

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

def plot_fft(data, plot):
    L = len(data)  # Chiều dài dữ liệu
    Y = np.fft.fft(data)  # Áp dụng biến đổi Fourier nhanh (FFT) lên dữ liệu
    Y[0] = 0  # Đặt thành phần tần số 0 bằng 0 để loại bỏ thành phần DC (1 chiều)
    P2 = np.abs(Y / L)  # Tính phổ mạch cao bằng giá trị tuyệt đối của FFT chia cho chiều dài dữ liệu
    P1 = P2[:L // 2 + 1]  # Lấy nửa phổ (từ 0 đến fs/2)
    P1[1:-1] = 2 * P1[1:-1]  # Nhân đôi các giá trị phổ (trừ điểm đầu và cuối)
    f1 = np.arange(len(P1)) * fs / len(P1) / 2  # Tạo mảng tần số f1
    if plot == 1:
        plt.plot(f1, P1)  # Vẽ đồ thị phổ
        plt.xlim(0, 50)  # Đặt giới hạn trục x trong khoảng 0 đến 100
        plt.show()  # Hiển thị đồ thị
    return P1

P1 = plot_fft(data, plot=1)  # Tính và vẽ đồ thị FFT của dữ liệu

f1 = np.arange(len(P1)) * fs / len(P1)  # Tạo mảng tần số f1
indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]  # Tìm chỉ mục của phổ trong khoảng delta (0.5 Hz - 4 Hz)
delta = np.sum(P1[indices1])  # Tính tổng phổ trong khoảng delta

f1 = np.arange(len(P1)) * fs / len(P1)  # Tạo mảng tần số f1
indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]  # Tìm chỉ mục của phổ trong khoảng theta (4 Hz - 8 Hz)
theta = np.sum(P1[indices1])  # Tính tổng phổ trong khoảng theta

f1 = np.arange(len(P1)) * fs / len(P1)  # Tạo mảng tần số f1
indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]  # Tìm chỉ mục của phổ trong khoảng alpha (8 Hz - 13 Hz)
alpha = np.sum(P1[indices1])  # Tính tổng phổ trong khoảng alpha

f1 = np.arange(len(P1)) * fs / len(P1)  # Tạo mảng tần số f1
indices1 = np.where((f1 >= 13) & (f1 <= 40))[0]  # Tìm chỉ mục của phổ trong khoảng beta (13 Hz - 40 Hz)
beta = np.sum(P1[indices1])  # Tính tổng phổ trong khoảng beta

f1 = np.arange(len(P1)) * fs / len(P1)  # Tạo mảng tần số f1
indices1 = np.where((f1 > 40) & (f1 <= 100))[0]  # Tìm chỉ mục của phổ trong khoảng gamma (40 Hz - 100 Hz)
gamma = np.sum(P1[indices1])  # Tính tổng phổ trong khoảng gamma

plt.bar('delta', delta)  # Vẽ biểu đồ cột cho tổng delta
plt.bar('theta', theta)  # Vẽ biểu đồ cột cho tổng theta
plt.bar('alpha', alpha)  # Vẽ biểu đồ cột cho tổng alpha
plt.bar('beta', beta)  # Vẽ biểu đồ cột cho tổng beta
plt.bar('gamma', gamma)  # Vẽ biểu đồ cột cho tổng gamma
plt.show()  # Hiển thị biểu đồ cột

data = filter_data(data)  # Lọc dữ liệu
plt.plot(data, linewidth=0.1)  # Vẽ đồ thị dữ liệu đã lọc
plt.ylim(-256,256)
plt.show()  # Hiển thị đồ thị

f, t, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)  # Áp dụng biến đổi Fourier ngắn hạn (STFT) lên dữ liệu
plt.pcolormesh(t, f, np.abs(Zxx), vmin=-1, vmax=5, shading='auto')  # Vẽ đồ thị màu dựa trên biên độ của STFT
plt.ylim(0.5, 40)  # Đặt giới hạn trục y trong khoảng 0.5 đến 40
plt.show()  # Hiển thị đồ thị
