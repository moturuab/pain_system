import tkinter as tk
import tkinter.font as fnt
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import time
from tkinter import filedialog
import glob
import datetime
import pain_detector
from pain_detector import *
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import itertools
import faulthandler
# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.message import EmailMessage

faulthandler.enable()

class VideoApp:
    def __init__(self, window, window_title, model_location):
        #self.send_email()
        self.window = window
        self.window.title(window_title)
        self.model_location = model_location

        # self.pain_detector = PainDetector(image_size=160,
        #                             checkpoint_path='/project/6005917/moturuab/dementia-pain/pain_detection_demo/checkpoints/50342566/50343918_3/model_epoch4.pt',
        #                             num_outputs=40)
        self.pain_detector = PainDetector(image_size=160,
                                          checkpoint_path=self.model_location,
                                          num_outputs=40)

        self.reference_images = []

        self.video_source = 1
        self.vid = cv2.VideoCapture(self.video_source)

        self.photo_frame = tk.Frame(window)
        self.photo_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.text_frame = tk.Frame(window)
        self.text_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.buttons_frame = tk.Frame(window)
        self.buttons_frame.pack(side=tk.RIGHT, padx=100, pady=100)

        img_label = tk.Label(self.photo_frame, image=None, compound="left")
        img_label.image = None
        img_label.pack(fill=tk.BOTH, side=tk.LEFT, padx=50, pady=50)

        font = fnt.Font(size=16)

        self.btn_stop = tk.Button(self.buttons_frame, text="Stop Session", font=font, width=20, height=4,
                                  command=self.stop_video)
        self.btn_stop.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_start = tk.Button(self.buttons_frame, text="Start Session", font=font, width=20, height=4,
                                   command=self.start_video)
        self.btn_start.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_take_image = tk.Button(self.buttons_frame, text="Take Reference Image", font=font, width=20, height=4,
                                        command=self.take_reference_image)
        self.btn_take_image.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_select_participant = tk.Button(self.buttons_frame, text="Select Participant", font=font, width=20,
                                                height=4, command=self.select_participant)
        self.btn_select_participant.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.font = fnt.Font(size=27)

        self.text = tk.Label(window, text='Please select a participant.', font=self.font)
        self.text.pack(fill=tk.BOTH, padx=10, pady=55, side=tk.BOTTOM)

        self.participant_label = tk.Label(self.text_frame, text=' ', font=self.font)
        self.participant_label.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.TOP)

        self.participant_number = 0

        self.btn_take_image["state"] = "disabled"
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "disabled"

        self.canvas.pack(fill=tk.BOTH, side=tk.LEFT)
        self.running = False

        self.MAX_FRAMES = 250  # 20 (20 seconds) * 2 (before and after the pain moment) * 15 (at 15 frames per second)
        self.threshold = 0.3

        self.pain_scores = deque(maxlen=self.MAX_FRAMES)
        self.frames = deque(maxlen=self.MAX_FRAMES)
        self.count = deque(maxlen=self.MAX_FRAMES)
        self.indices = deque(maxlen=self.MAX_FRAMES)
        self.times = deque(maxlen=self.MAX_FRAMES)

        self.start_index = 0
        self.end_index = 0
        self.index = 0
        self.time = 0
        self.after_pain_count = 0
        self.pain_moment = False

        self.lock = threading.Lock()

        self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()

    def start_video(self):
        self.running = True
        self.text.config(text='Session ongoing.')
        self.model_thread = threading.Thread(target=self.run_model, daemon=True)
        self.model_thread.start()
        self.btn_stop["state"] = "normal"
        self.btn_start["state"] = "disabled"
        self.start_time = datetime.datetime.now()
        self.log_entry(str(self.participant_number) + ',' + self.start_time.strftime("%Y-%m-%d+%H:%M:%S.%f")[:-3])

    def turn_on_light(self):
        pass

    def send_email(self):
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()

        # authentication
        s.login("abhi.saim@gmail.com", "orut ntbi xxjq hrqm")

        # create email message
        msg = EmailMessage()
        msg['Subject'] = 'Vision System Alert: Participant ' + str(self.participant_number)
        msg['From'] = 'abhi.saim@gmail.com'
        msg['To'] = 'abhishek.moturu@mail.utoronto.ca'
        msg.set_content('Please check on Participant ' + str(self.participant_number) +
                        ' as a suspected pain expression has been detected.')

        # sending the mail
        s.send_message(msg)

        # terminating the session
        s.quit()

    def log_entry(self, entry, end=''):
        f = open("system_log.txt", "a")
        f.write(entry + end)
        f.close()

    def run_model(self):
        while self.running:
            with (self.lock):
                self.indices.append(self.index)
                pain_score = self.pain_detector.predict_pain(self.frame)
                self.pain_scores.append(pain_score)

                if not self.pain_moment and pain_score > self.threshold:
                    self.pain_moment = True
                    self.start_index = self.index
                    self.turn_on_light()
                    self.send_email()
                    self.text.config(text='Session ongoing: suspected pain expression.')

                if self.pain_moment and (self.end_index - self.start_index < self.MAX_FRAMES/2 or self.indices[-1] - self.indices[0] < self.MAX_FRAMES):
                    self.end_index = self.index

                elif self.pain_moment and self.end_index - self.start_index >= self.MAX_FRAMES/2 and self.indices[-1] - self.indices[0] >= self.MAX_FRAMES:
                    for c in self.count:
                        try:
                            i = self.indices.index(c)
                            i_ = self.count.index(c)
                        except ValueError:
                            continue
                        break
                    j = self.indices.index(self.end_index)+1
                    j_ = self.count.index(self.end_index)+1
                    self.save_video(deque(itertools.islice(self.frames, i_, j_)), deque(itertools.islice(self.pain_scores, i, j)), deque(itertools.islice(self.indices, i, j)), deque(itertools.islice(self.times, i, j)))
                    self.after_pain_count = 0
                    self.pain_moment = False

    def stop_video(self):
        self.running = False
        self.btn_start["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        self.text.config(text='Please start the session when ready.')
        self.end_time = datetime.datetime.now()
        self.index = 0
        difference = (self.end_time - self.start_time).total_seconds()/60.0
        self.log_entry(',' + self.end_time.strftime("%Y-%m-%d+%H:%M:%S.%f")[:-3] + ',' + str(round(difference, 3)), '\n')

    def save_video(self, frames, pain_scores, indices, times):
        height, width, layers = frames[0].shape

        video = cv2.VideoWriter(os.path.join(self.participant, 'video_' +
                                             str(len(glob.glob1(self.participant, "*.mp4")) + 1) + '.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

        self.text.config(text='Session ongoing: saving video.')

        for frame in list(frames):
            video.write(frame)

        frms = [i for i in indices]
        inds = [i-indices[0] for i in indices]
        times = [i for i in times]

        cv2.destroyAllWindows()
        video.release()

        self.text.config(text='Session ongoing.')

        with open(os.path.join(self.participant,
                               'pain_scores_' + str(len(glob.glob1(self.participant, "*.csv")) + 1) + '.csv'),
                  'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['time', 'index', 'frame', 'pain_score'])
            writer.writerows(zip(times, inds, frms, pain_scores))

    def update_frame(self):
        while True:
            ret, self.frame = self.vid.read()
            self.frame = cv2.putText(self.frame, str(self.index),(20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                     1, (0, 255, 255),2, cv2.LINE_4)
            if self.running:
                self.frames.append(self.frame)
                self.count.append(self.index)
                self.times.append(datetime.datetime.now().strftime("%Y-%m-%d+%H:%M:%S.%f")[:-3])
                self.index += 1
            frame = cv2.resize(self.frame, (int(5 * self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) // 7),
                                            int(5 * self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) // 7)))

            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(10, 10, image=self.photo, anchor=tk.NW)
            self.window.update()

    def select_participant(self):
        self.reference_images = []
        self.pain_detector.ref_frames = []
        self.participant_label.config(text='')
        for widget in self.photo_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.destroy()
        self.participant = filedialog.askdirectory(initialdir=os.getcwd() + '/pain_detection_demo/pain_data_collection/location_1/')
        for (root_, dirs, files) in os.walk(self.participant):
            if files:
                for file in files:
                    if file.endswith('.jpg'):
                        path = os.path.join(self.participant, file)
                        image = Image.open(path)
                        self.pain_detector.add_references([np.asarray(image)])
                        aspect_ratio = image.width / image.height
                        image = image.resize((int(100 * aspect_ratio), 100))
                        photo = ImageTk.PhotoImage(image)
                        img_label = tk.Label(self.photo_frame, image=photo, compound="left")
                        img_label.image = photo
                        img_label.pack(fill=tk.BOTH, side=tk.LEFT)
                        self.reference_images.append(photo)
        if len(self.reference_images) == 3:
            self.btn_start["state"] = "normal"
            self.text.config(text='Please start the session when ready.')
        elif len(self.reference_images) < 3 and 'participant' in self.participant:
            self.btn_take_image["state"] = "normal"
            self.text.config(text='Please take 3 reference images.')
        self.participant_number = int(self.participant.split('_')[-1])
        self.participant_label.config(text='Participant ' + str(self.participant_number))
        self.window.update()

    def take_reference_image(self):
        if len(self.reference_images) < 3:
            ret, frame = self.vid.read()
            if ret:
                path = os.path.join(self.participant,
                                    'reference_image_' + str(len(glob.glob1(self.participant, "*.jpg")) + 1) + '.jpg')

                self.pain_detector.add_references([frame])
                # prepped_image = self.pain_detector.ref_frames[-1]
                cv2.imwrite(path, frame)
                image = Image.open(path)
                aspect_ratio = image.width / image.height
                image = image.resize((int(100 * aspect_ratio), 100))
                photo = ImageTk.PhotoImage(image)
                img_label = tk.Label(self.photo_frame, image=photo, compound="left")
                img_label.image = photo
                img_label.pack(fill=tk.BOTH, side=tk.LEFT)
                self.reference_images.append(photo)
                if len(self.reference_images) == 3:
                    self.btn_start["state"] = "normal"
                    self.btn_take_image["state"] = "disabled"
                    self.text.config(text='Please start the session when ready.')
                self.window.update()

    def __del__(self):
        if self.vid.isOpened():
            self.video_thread.join()
            self.model_thread.join()
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(width=True, height=True)
    location = "Test"
    model = "/Users/abhishekmoturu/Desktop/dementia-pain/pain_detection_demo/checkpoints/50342566/50343918_3/model_epoch4.pt"
    app = VideoApp(root, location + " Vision System", model)
    root.mainloop()
