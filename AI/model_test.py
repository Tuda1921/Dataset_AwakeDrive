import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open('Gradient_1.pkl', 'rb') as file:
    model = pickle.load(file)

# X = pd.read_csv('..\Process\CSV\Feature_full_label.csv')
# print('Load_csv: Done')
# y = X['label']
# X = X.drop('label', axis=1)
# X = X.values

task1_test = np.loadtxt(r'D:\Tuda\OneDrive - Hanoi University of Science and Technology\Documents\Research\Awake Drive\Dataset_AwakeDrive\AD_UI\Data\Task1_cut\task1.txt')
from Process.processData import slide, filter_data
task1_test = pd.DataFrame(slide(filter_data(task1_test)))
# test_data = model.predict(np.array(task1_test))
prob_check = model.predict_proba(np.array(task1_test))
out = [0, 25, 50, 75, 100]
prob_check = np.dot(prob_check, out)
# from scipy.ndimage import gaussian_filter
# prob_check = gaussian_filter(prob_check, sigma=5)
# # print(test_data)
# # print(type(test_data))
plt.plot(prob_check)
plt.show()

