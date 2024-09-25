import tkinter as tk
import time
import random
import collectData 
import os
import threading
import numpy as np
import sounddevice as sd

class ExperimentApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Thí nghiệm Sóng não")
        self.root.geometry("1200x800")
        # self.root.bind('<space>', self.on_spacebar_press)
        self.root.bind('<Return>', self.on_enter_press)
        
        self.current_frame = None
        
        self.sound = Sound(frequency = 1000, duration = 0.4, fs = 44100)

        self.create_login_screen()
    
    def on_spacebar_press(self, event):
        if self.current_frame:
            # Gọi hàm tiếp theo dựa trên giao diện hiện tại
            if hasattr(self, 'next_function'):
                self.next_function()
                print("Spacebar pressed, moving to next function")
        # self.root.unbind('<space>')
        print("test")

    def on_enter_press(self, event):
        if self.current_frame:
            self.logged()
        self.root.unbind('<Return>')

    def create_login_screen(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Đăng nhập").pack()
        
        self.name_var = tk.StringVar()
        tk.Label(frame, text="Tên:").pack()
        tk.Entry(frame, textvariable=self.name_var).pack()

        self.age_var = tk.StringVar()
        tk.Label(frame, text="Tuổi:").pack()
        tk.Entry(frame, textvariable=self.age_var).pack()
        
        self.gender_var = tk.StringVar()
        tk.Label(frame, text="Giới tính:").pack()
        tk.Entry(frame, textvariable=self.gender_var).pack()
        
        self.address_var = tk.StringVar()
        tk.Label(frame, text="Địa chỉ:").pack()
        tk.Entry(frame, textvariable=self.address_var).pack()
        
        self.phone_var = tk.StringVar()
        tk.Label(frame, text="Số điện thoại:").pack()
        tk.Entry(frame, textvariable=self.phone_var).pack()
        
        tk.Button(frame, text="Đăng nhập", command=self.create_intro_screen).pack(pady=10)
        
        # if not os.path.exists(f"Data/{self.name_var.get()}"):
        # os.makedirs(f"Data/{self.name_var.get()}", exist_ok=True)

        self.logged = self.create_intro_screen
        self.current_frame = frame

    def create_intro_screen(self):
        self.clear_frame()
        self.root.bind('<space>', self.on_spacebar_press)
        
        print(self.name_var.get(), self.age_var.get())

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Giới thiệu về dự án").pack()
        tk.Label(frame, text="Chào mừng bạn đến với dự án XYZ. Hãy bấm phím [spacebar] để tiếp tục.").pack(pady=20)

        self.next_function = self.create_general_instructions
        self.current_frame = frame
    
    def create_general_instructions(self):
        self.clear_frame()
        self.root.unbind('<space>')

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Hướng dẫn chung").pack()
        tk.Label(frame, text="Trước khi bắt đầu, hãy đọc kỹ hướng dẫn sau đây. Bấm phím [spacebar] để tiếp tục.").pack(pady=20)

        self.next_function = self.create_task1_instructions
        self.root.bind('<space>', self.on_spacebar_press)
        print("test")
        self.current_frame = frame
    
    def create_task1_instructions(self):
        self.clear_frame()
        self.root.unbind('<space>')

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 1 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 1. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)

        self.next_function = self.create_task1_execution
        self.root.bind('<space>', self.on_spacebar_press)
        self.current_frame = frame

    def create_task1_execution(self):
        self.clear_frame()
        self.root.unbind('<space>')
        self.sound.playSound()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 1 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trắng\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)

        canvas = tk.Canvas(frame, width=800, height=600, bg="white")
        canvas.pack()
        
        self.task1_timer = tk.StringVar()
        self.task1_timer.set("Thời gian: 0:05")

        tk.Label(frame, textvariable=self.task1_timer).pack()

        self.current_frame = frame

        current_time = self.task1_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var.get()}/task1.txt"
        data_thread = threading.Thread(target=collectData.collectData, args=(path, time_rec, port))
        data_thread.start()

        self.root.after(1000, self.update_task1_timer)

    def update_task1_timer(self):
        current_time = self.task1_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task1_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 1. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            start = time.time()
            self.sound.playSound()
            end = time.time()
            print(end - start)
            self.next_function = self.create_task2_instructions
            self.root.bind('<space>', self.on_spacebar_press)
        else:
            self.root.after(1000, self.update_task1_timer)

    def create_task2_instructions(self):
        self.clear_frame()
        self.root.unbind('<space>')

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 2 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 2. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)

        self.next_function = self.create_task2_execution
        self.root.bind('<space>', self.on_spacebar_press)
        self.current_frame = frame

    def create_task2_execution(self):
        self.clear_frame()
        self.root.unbind('<space>')
        self.sound.playSound()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 2 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trắng\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)
        
        canvas = tk.Canvas(frame, width=800, height=600, bg="white")
        canvas.pack()
        
        self.task2_timer = tk.StringVar()
        self.task2_timer.set("Thời gian: 2:30")

        current_time = self.task2_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var.get()}/task2.txt"
        data_thread = threading.Thread(target=collectData.collectData, args=(path, time_rec, port))
        data_thread.start()

        tk.Label(frame, textvariable=self.task2_timer).pack()

        self.current_frame = frame

        self.root.after(1000, self.update_task2_timer)

    def update_task2_timer(self):
        current_time = self.task2_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task2_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 2. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            self.sound.playSound()
            self.root.bind('<space>', self.on_spacebar_press)
            self.next_function = self.create_task3_instructions
        else:
            self.root.after(1000, self.update_task2_timer)

    def create_task3_instructions(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 3 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 3. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)

        self.next_function = self.create_task3_execution
        self.root.bind('<space>', self.on_spacebar_press)
        self.current_frame = frame

    def create_task3_execution(self):
        self.clear_frame()
        self.root.unbind('<space>')
        self.sound.playSound()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 3 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trắng\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)

        canvas = tk.Canvas(frame, width=800, height=600, bg="white")
        canvas.pack()
        
        self.task3_timer = tk.StringVar()
        self.task3_timer.set("Thời gian: 2:30")

        current_time = self.task3_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var.get()}/task3.txt"
        data_thread = threading.Thread(target=collectData.collectData, args=(path, time_rec, port))
        data_thread.start()

        tk.Label(frame, textvariable=self.task3_timer).pack()

        self.current_frame = frame

        self.root.after(1000, self.update_task3_timer)

    def update_task3_timer(self):
        current_time = self.task3_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task3_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 3. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            self.root.bind('<space>', self.on_spacebar_press)
            self.sound.playSound()
            self.next_function = self.create_task4_instructions
        else:
            self.root.after(1000, self.update_task3_timer)

    def create_task4_instructions(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 4 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 4. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)

        self.next_function = self.create_task4_execution
        self.root.bind('<space>', self.on_spacebar_press)
        self.current_frame = frame

    def create_task4_execution(self):
        self.clear_frame()
        self.root.unbind('<space>')
        self.sound.playSound()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 4 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trắng với chấm tròn đỏ ở giữa\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)

        canvas = tk.Canvas(frame, width=800, height=600, bg="white")
        canvas.pack()
        
        # Vẽ chấm tròn đỏ ở giữa
        center_x = 400
        center_y = 300
        radius = 10
        canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill="red")
        
        self.task4_timer = tk.StringVar()
        self.task4_timer.set("Thời gian: 2:30")

        current_time = self.task4_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var.get()}/task4.txt"
        data_thread = threading.Thread(target=collectData.collectData, args=(path, time_rec, port))
        data_thread.start()

        tk.Label(frame, textvariable=self.task4_timer).pack()

        self.current_frame = frame

        self.root.after(1000, self.update_task4_timer)

    def update_task4_timer(self):
        current_time = self.task4_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task4_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 4. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            self.root.bind('<space>', self.on_spacebar_press)
            self.sound.playSound()
            self.next_function = self.create_task5_instructions
        else:
            self.root.after(1000, self.update_task4_timer)

    def create_task5_instructions(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 5 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 5. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)
        # tk.Button(frame, text="Sẵn sàng", command=self.create_task5_execution).pack()

        self.current_frame = frame
        self.next_function = self.create_task5_execution
        self.root.bind('<space>', self.on_spacebar_press)

    def create_task5_execution(self):
        self.clear_frame()
        self.root.unbind('<space>')
        self.sound.playSound()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 5 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trò chơi\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)

        self.task5_timer = tk.StringVar()
        self.task5_timer.set("Thời gian: 2:30")

        current_time = self.task5_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var.get()}/task5.txt"
        data_thread = threading.Thread(target=collectData.collectData, args=(path, time_rec, port))
        data_thread.start()

        tk.Label(frame, textvariable=self.task5_timer).pack()

        self.score_label = tk.StringVar()
        self.score_label.set("Điểm: 0")
        tk.Label(frame, textvariable=self.score_label).pack()

        canvas = tk.Canvas(frame, width=400, height=400, bg="white")
        canvas.pack()

        self.arrow = Arrow(canvas, self.score_label)
        self.arrow.draw()

        self.start_time = time.time()
        
        self.root.bind("<Up>", self.arrow.handle_keypress)
        self.root.bind("<Down>", self.arrow.handle_keypress)
        self.root.bind("<Left>", self.arrow.handle_keypress)
        self.root.bind("<Right>", self.arrow.handle_keypress)

        self.current_frame = frame

        self.root.after(1000, self.update_task5_timer)

    def update_task5_timer(self):
        current_time = self.task5_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task5_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            self.end_time = time.time()
            reaction_time = (self.end_time - self.start_time) / max(1, self.arrow.attempts)
            self.show_results(reaction_time)

            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 5. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            self.root.bind('<space>', self.on_spacebar_press)
            self.sound.playSound()
            self.next_function = self.create_end_screen

        else:
            self.root.after(1000, self.update_task5_timer)

    def show_results(self, reaction_time):
        self.root.unbind("<Up>")
        self.root.unbind("<Down>")
        self.root.unbind("<Left>")
        self.root.unbind("<Right>")
        self.arrow.clear()
        self.arrow.show_results(self.arrow.score, reaction_time)

    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

    def create_end_screen(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Kết thúc quá trình đo thí nghiệm").pack()
        tk.Label(frame, text="Cảm ơn bạn đã tham gia. Các file dữ liệu đã đo được liệt kê dưới đây.\nCác file dữ liệu đã nằm trong folder Data -> {}".format(self.name_var.get())).pack(pady=20)
        
        files = ["data_{}_{}.txt".format(self.name_var.get(), self.age_var.get())]  # Example filename
        for file in files:
            tk.Label(frame, text=file).pack()

        self.current_frame = frame

class Arrow:
    def __init__(self, canvas, score_label):
        self.canvas = canvas
        self.directions = [
            {"coords": [(25,0), (0,50), (25,40), (50,50)], "name": "up"},
            {"coords": [(0,25), (50,0), (40,25), (50,50)], "name": "left"},
            {"coords": [(25,50), (0,0), (25,10), (50,0)], "name": "down"},
            {"coords": [(50,25), (0,0), (10,25), (0,50)], "name": "right"}
        ]
        self.current_direction = None
        self.arrow_id = None
        self.score = 0
        self.attempts = 0
        self.score_label = score_label

    def draw(self):
        if self.arrow_id:
            self.canvas.delete(self.arrow_id)
        self.current_direction = random.choice(self.directions)
        self.arrow_id = self.canvas.create_polygon(
            self.current_direction["coords"], fill="black"
        )
        self.canvas.coords(self.arrow_id, sum([(x + 175, y + 175) for x, y in self.current_direction["coords"]], ()))

    def check_direction(self, key):
        if key == "Up" and self.current_direction["name"] == "up":
            return True
        elif key == "Down" and self.current_direction["name"] == "down":
            return True
        elif key == "Left" and self.current_direction["name"] == "left":
            return True
        elif key == "Right" and self.current_direction["name"] == "right":
            return True
        return False

    def handle_keypress(self, event):
        self.attempts += 1
        if self.check_direction(event.keysym):
            self.score += 1
            self.canvas.itemconfig(self.arrow_id, fill="green")
        else:
            self.canvas.itemconfig(self.arrow_id, fill="red")
        self.score_label.set(f"Điểm: {self.score}")
        self.canvas.after(200, self.draw)

    def clear(self):
        if self.arrow_id:
            self.canvas.delete(self.arrow_id)

    def show_results(self, score, reaction_time):
        self.canvas.create_text(200, 150, text=f"Tổng điểm của bạn là: {score}", font=("Helvetica", 16), fill="black")
        self.canvas.create_text(200, 200, text=f"Thời gian phản ứng trung bình của bạn là: {reaction_time:.2f} giây", font=("Helvetica", 16), fill="black")

class Sound:
    def __init__(self, frequency, duration, fs):
        self.frequency = frequency
        self.duration = duration
        self.fs = fs
    def createWave(self):
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.wave = 0.5 * np.sin(2 * np.pi * self.frequency * t)
    def playSound(self):
        self.createWave()
        sd.play(self.wave, self.fs)
        sd.wait()

if __name__ == "__main__":
    port = "COM5"
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
