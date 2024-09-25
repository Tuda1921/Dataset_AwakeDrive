# from PIL import Image, ImageTk
# import tkinter as tk
# import time
# import sys
# import os
# import tensorflow as tf
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import serial

# if serial.Serial:
#     serial.Serial().close()

# from Process.processData import slide, filter_data, FeatureExtract

# with open('Gradient_1.pkl', 'rb') as file:
#     model = pickle.load(file)


# s = serial.Serial("COM3", baudrate=57600)
# print("START!")


# def plotData(data, features, attention_score):
#     fig = plt.figure(figsize=(10, 5))
#     # Plot raw
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax1.plot(data)
#     ax1.set_title('EEG Raw Values')
#     ax1.set_xlabel('Samples')
#     ax1.set_ylabel('RawValue')
#     ax1.set_ylim(-256, 256)

#     # Plot STFT
#     ax2 = fig.add_subplot(1, 2, 2)
#     ax2.plot(attention_score)
#     ax2.set_title("Attention Score")
#     ax2.set_xlabel('Time [sec]')
#     ax2.set_ylabel('Score')
#     ax2.set_ylim(0, 100)

#     # Plot brainwave
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax3.plot(features['delta'], label="delta")
#     ax3.plot(features['theta'], label="theta")
#     ax3.plot(features['alpha'], label="alpha")
#     ax3.plot(features['beta'], label="beta")
#     ax3.set_title('Frequency Bands')
#     ax3.set_xlabel('Time [sec]')
#     ax3.set_ylabel('Power')
#     ax3.set_ylim(0, 400)
#     ax3.legend()

#     # Hiển thị hình ảnh
#     plt.tight_layout()
#     plt.savefig("test.png")
#     plt.close()


# def start_test():
#     k = 15*512
#     y = np.array([], dtype=int)
#     x = 0

#     while x < (3600 * 512):
#         try:
#             # if x % 512 == 0:
#             #     print(x//512)
#             data = s.readline().decode('utf-8').rstrip("\r\n")  # strip removes leading/trailing whitespace
#             x += 1
#             y = np.append(int(data))

#             if x >= k:
#                 if x % (1 * 512) == 0:
#                     sliding_window_start = x - k
#                     sliding_window_end = x
#                     sliding_window = np.array(y[sliding_window_start:sliding_window_end])
#                     features = FeatureExtract(sliding_window)
#                     feature_test = np.array(list(features.values())).reshape(1,-1)
#                     print(model.predict(feature_test))

#                     prob_check = model.predict_proba(np.array(feature_test))
#                     score = [0, 25, 50, 75, 100]

#                     attention_score = np.dot(prob_check, score)
#                     print(attention_score)

#                     plotData(sliding_window, features, attention_score)
#                     show_image()

#         except Exception as e:
#             print(f"Đã xảy ra lỗi: {e}")

#     np.savetxt("demo.txt", y, fmt="%d")


# # Use to get data txt


# image_path1 = 'test.png'

# status_label = None

# def show_image():
#     image1 = Image.open(image_path1)
#     photo1 = ImageTk.PhotoImage(image1)
#     image_frame1.configure(image=photo1)
#     window.update()
#     image1.close()


# window = tk.Tk()

# window.title("Awake Drive")

# isPause = tk.IntVar()
# pause_button = tk.Checkbutton(window, text="Stop", variable=isPause, onvalue=1, offvalue=0)
# pause_button.pack(pady=10)

# start_button = tk.Button(window, text="Start", command=start_test)
# start_button.pack(pady=10)
# # Raw wave
# image_frame1 = tk.Label(window, width=1000, height=500, bg="white")
# image_frame1.pack(side=tk.LEFT)

# show_image()
# # Tạo một khung văn bản để hiển thị trạng thái buồn ngủ/tỉnh táo


# window.mainloop()
# # Close the serial port
# print("DONE")
# # s.close()
# # file.close()


from PIL import Image, ImageTk
import tkinter as tk
import time
import sys
import os
# import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
from sklearn.ensemble import GradientBoostingClassifier
from collections import deque

if serial.Serial:
    serial.Serial().close()

with open('Gradient_1.pkl', 'rb') as file:
    model = pickle.load(file)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Process.processData import slide, filter_data, FeatureExtract

s = serial.Serial("COM3", baudrate=57600)
# s = open("/Users/nguyentrithanh/Documents/Lab/EEG_AwakeDrive/test/Data/Dat1/task2.txt", 'rb')
print("START!")


def plotData(data, features, attention_score):
    fig = plt.figure(figsize=(10, 5))
    # Plot raw
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data)
    ax1.set_title('EEG Raw Values')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('RawValue')
    ax1.set_ylim(-256, 256)

    # Plot STFT
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(attention_score)
    ax2.set_title("Attention Score")
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 100)

    # Plot brainwave
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(features[0], label="delta")
    ax3.plot(features[1], label="theta")
    ax3.plot(features[2], label="alpha")
    ax3.plot(features[3], label="beta")
    ax3.set_title('Frequency Bands')
    ax3.set_xlabel('Time [sec]')
    ax3.set_ylabel('Power')
    ax3.set_ylim(0, 400)
    ax3.legend()

    # Hiển thị hình ảnh
    plt.tight_layout()
    plt.savefig("test.png")
    plt.close()


def start_test():
    k = 15 * 512
    y = np.array([], dtype=int)
    x = 0
    slide_length = 15
    features_deque = [deque(maxlen=slide_length) for _ in range(4)]
    attention_score_deque = deque(maxlen=slide_length)

    while x < (3600 * 512):
        # try:
        if x % 512 == 0:
            print(x // 512)
        data = s.readline().decode('utf-8').rstrip("\r\n")  # strip removes leading/trailing whitespace
        # data = s.readline()
        print('check')
        x += 1
        y = np.append(y, int(data))

        # if x >= k:
        if x % (1 * 512) == 0:
            sliding_window_start = x - k
            sliding_window_end = x
            sliding_window = np.array(y[sliding_window_start:sliding_window_end])
            features = FeatureExtract(sliding_window)

            features_deque[0].append(features['delta'])
            features_deque[1].append(features['theta'])
            features_deque[2].append(features['alpha'])
            features_deque[3].append(features['beta'])

            feature_test = np.array(list(features.values())).reshape(1, -1)
            print(model.predict(feature_test))

            prob_check = model.predict_proba(np.array(feature_test))
            score = [0, 25, 50, 75, 100]

            attention_score = np.dot(prob_check, score)
            attention_score_deque.append(attention_score)
            print(attention_score)

            plotData(sliding_window, features_deque, attention_score_deque)
            show_image()
            time.sleep(0.5)
        # except Exception as e:
        #     print(f"Đã xảy ra lỗi: {e}")

    np.savetxt("demo.txt", y, fmt="%d")


# Use to get data txt


image_path1 = 'test.png'

status_label = None


def show_image():
    image1 = Image.open(image_path1)
    photo1 = ImageTk.PhotoImage(image1)
    image_frame1.configure(image=photo1)
    window.update()
    image1.close()


window = tk.Tk()

window.title("Awake Drive")

isPause = tk.IntVar()
pause_button = tk.Checkbutton(window, text="Stop", variable=isPause, onvalue=1, offvalue=0)
pause_button.pack(pady=10)

start_button = tk.Button(window, text="Start", command=start_test)
start_button.pack(pady=10)
# Raw wave
image_frame1 = tk.Label(window, width=1000, height=500, bg="white")
image_frame1.pack(side=tk.LEFT)

show_image()
# Tạo một khung văn bản để hiển thị trạng thái buồn ngủ/tỉnh táo


window.mainloop()
# Close the serial port
print("DONE")
# s.close()
# file.close()


