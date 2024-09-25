import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial

if serial.Serial:
    serial.Serial().close()

with open('Gradient_1.pkl', 'rb') as file:
    model = pickle.load(file)

# X = pd.read_csv('..\Process\CSV\Feature_full_label.csv')
# print('Load_csv: Done')
# y = X['label']
# X = X.drop('label', axis=1)
# X = X.values

x = 0
y = []
s = serial.Serial("COM3", baudrate=57600)
print("START!")
k = 15*512
from Process.processData import slide, filter_data, FeatureExtract
while x < (3600 * 512):
    # try:
        # if x % 512 == 0:
        #     print(x//512)
        data = s.readline().decode('utf-8').rstrip("\r\n")  # strip removes leading/trailing whitespace
        x += 1
        y.append(int(data))

        if x >= k:
            if x % (1 * 512) == 0:
                sliding_window_start = x - k
                sliding_window_end = x
                sliding_window = np.array(y[sliding_window_start:sliding_window_end])

                # feature.append(FeatureExtract(sliding_window)) #abc
                feature_test = np.array(list(FeatureExtract(sliding_window).values())).reshape(1,-1)
                # print(feature_test)
                print(model.predict(feature_test))
                # show_image()
                sliding_window = []

                prob_check = model.predict_proba(np.array(feature_test))
                print(prob_check)
                out = [0, 25, 50, 75, 100]
                prob_check = np.dot(prob_check, out)
                print(prob_check)
                # # print(test_data)
                # # print(type(test_data))
                # plt.plot(prob_check)
                # plt.show()
    # except Exception as e:
    #     print(f"Đã xảy ra lỗi: {e}")


