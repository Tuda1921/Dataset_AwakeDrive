import numpy as np
import serial
import os

def collectData(path, time_rec, port):
    if serial.Serial:
        serial.Serial().close()
    # Open the serial port
    s = serial.Serial(port, baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.

    dir_name = os.path.dirname(path)
    
    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file = open(path, "w")

    x = 0  # iterator of sample
    while True:
        try:
            data = s.readline().decode('utf-8').rstrip("\r\n")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

    print("START!")
    while x < (time_rec * 512):
        try:
            if x % 512 == 0:
                print(x // 512)
            x += 1
            data = s.readline().decode('utf-8').rstrip("\r\n")
            file.write(str(data))
            file.write('\n')
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
    # Close the serial port
    print("DONE")
    s.close()
    file.close()
    return 

if __name__ == "__main__":
    collectData("Task1/Test_Thanh/task1.txt", 36000, "COM5")
