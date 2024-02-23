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
import pywemo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import itertools
import faulthandler
import mediapipe as mp
import smtplib
import argparse
import subprocess
from email.message import EmailMessage
from point_grey import PgCamera

if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

faulthandler.enable()

# cd Desktop\pain_system
# pain_env\Scripts\activate
# python pain_detection.py

parser = argparse.ArgumentParser(prog='Vision System', description='This is a vision system for monitoring pain.')
parser.add_argument('-threshold', type=float, default=0.2422)
parser.add_argument('-from_email', type=str, default='uofr.healthpsychologylab@gmail.com')
parser.add_argument('-to_email', type=str, default='uofr.healthpsychologylab@gmail.com')
parser.add_argument('-wemo_code', type=str, default='12E')
parser.add_argument('-ssd', type=str, default='G:\CentralHavenSaskatoon')
parser.add_argument('-seconds', type=int, default=3)
parser.add_argument('-high_frames', type=int, default=5)
arg_dict = parser.parse_args()

class VideoApp:
    def __init__(self, window, window_title, ssd_location, model_location, location, threshold, seconds, high_frames, ltch_wifi, wemo_wifi, from_email, to_emails):
        self.window = window
        self.window_title = window_title
        self.window.title(self.window_title)
        self.ssd_location = ssd_location
        self.model_location = model_location
        self.location = location
        self.threshold = threshold
        self.seconds = seconds
        self.high_frames = high_frames
        self.ltch_wifi = ltch_wifi
        self.wemo_wifi = wemo_wifi
        self.from_email = from_email
        self.to_emails = to_emails

        self.pain_detector = PainDetector(image_size=160, checkpoint_path=self.model_location, num_outputs=40)
        self.reference_images = []

        self.video_source = 0
        self.width = 1510
        self.height = 1080
        self.offset = 0
        roi = [self.offset, self.offset, self.width, self.height]
        frame_rate = 15.0
        gain = 34
        exposure_ms = 20.
        self.vid = PgCamera(roi, frame_rate, gain, exposure_ms)
        self.vid.open_camera()

        self.photo_frame = tk.Frame(window)
        self.photo_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.text_frame = tk.Frame(window)
        self.text_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.canvas = tk.Label(window, width=self.width, height=self.height)

        self.buttons_frame = tk.Frame(window)
        self.buttons_frame.pack(side=tk.RIGHT, padx=100, pady=50)

        img_label = tk.Label(self.photo_frame, image=None, compound="left")
        img_label.image = None
        img_label.pack(fill=tk.BOTH, side=tk.LEFT, padx=50, pady=50)

        font = fnt.Font(size=63)

        self.blank = tk.Label(self.buttons_frame, text=" ", font=font)
        self.blank.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        font = fnt.Font(size=16)
        self.btn_stop = tk.Button(self.buttons_frame, text="Stop Session", font=font, width=20, height=2,
                                  command=self.stop_video)
        self.btn_stop.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_light = tk.Button(self.buttons_frame, text="Checked Resident", font=font, width=20, height=2,
                                   command=self.turn_off_light)
        self.btn_light.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_start = tk.Button(self.buttons_frame, text="Start Session", font=font, width=20, height=2,
                                   command=self.start_video)
        self.btn_start.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_take_image = tk.Button(self.buttons_frame, text="Take Reference Image", font=font, width=20, height=2,
                                        command=self.take_reference_image)
        self.btn_take_image.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.btn_select_participant = tk.Button(self.buttons_frame, text="Select Participant", font=font, width=20,
                                                height=2, command=self.select_participant)
        self.btn_select_participant.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.BOTTOM)

        self.font = fnt.Font(size=27)

        self.text = tk.Label(window, text='LOADING...', font=self.font)
        self.text.pack(fill=tk.BOTH, padx=10, pady=55, side=tk.BOTTOM)

        self.participant_label = tk.Label(self.text_frame, text=' ', font=self.font)
        self.participant_label.pack(fill=tk.BOTH, padx=10, pady=10, side=tk.TOP)

        self.participant_number = 0

        self.btn_take_image["state"] = "disabled"
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "disabled"
        self.btn_light["state"] = "disabled"
        self.btn_select_participant["state"] = "disabled"

        self.canvas.pack(fill=tk.BOTH, side=tk.LEFT, padx=70)
        self.running = False

        self.MAX_FRAMES = 700  # 20 (20 seconds) * 2 (before and after pain) * 15 (at 15 fps) + 100 frames
        self.threshold = 0.3

        self.pain_scores = deque(maxlen=self.MAX_FRAMES)
        self.frames = deque(maxlen=self.MAX_FRAMES)
        self.count = deque(maxlen=self.MAX_FRAMES)
        self.indices = deque(maxlen=self.MAX_FRAMES)
        self.times = deque(maxlen=self.MAX_FRAMES)

        self.start_time = None
        self.end_time = None
        self.start_index = 0
        self.end_index = 0
        self.index = 0
        self.after_pain_count = 0
        self.pain_moment = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9)

        self.lock = threading.Lock()
        self.img = None

        self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()
        
    def connect_to_network(self, network_name):
        command = f'netsh wlan connect name={network_name} ssid={network_name}'
        subprocess.run(command, shell=True)
        return self.get_ssid(network_name)
        
    def get_ssid(self, network_name):
        data_strings = []
        while 'Profile' not in data_strings:
            raw_wifi = subprocess.check_output(['netsh', 'WLAN', 'show', 'interfaces'])
            data_strings = raw_wifi.decode('utf-8').split()
        index = data_strings.index('SSID')
        return data_strings[index + 2] == network_name.strip('"').strip("'").split()[0]

    def start_video(self):
        self.running = True
        self.text.config(text='Session ongoing.')
        self.model_thread = threading.Thread(target=self.run_model, daemon=True)
        self.model_thread.start()
        self.btn_stop["state"] = "normal"
        self.btn_start["state"] = "disabled"
        self.btn_select_participant["state"] = "disabled"
        self.start_time = datetime.datetime.now()
        self.log_entry('\n' + str(self.participant_number) + ',' + self.start_time.strftime("%b %d %Y %H:%M:%S.%f")[:-3], 'summary_log.txt')
        self.log_entry('----------\nParticipant ' + str(self.participant_number) + ' (threshold: ' + str(arg_dict.threshold) + ')\n', 'full_log.txt')
        self.log_entry('Started session: ' + self.start_time.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + '.\n', 'full_log.txt')

    def turn_on_light(self):
        while True:
            if self.connect_to_network(self.wemo_wifi):
                break
        while True:
            try:
                self.devices = pywemo.discover_devices()
                self.devices[0].on()
                self.email_thread = threading.Thread(target=self.send_email, daemon=True)
                self.email_thread.start()
                break
            except:
                continue

    def turn_off_light(self):
        time = datetime.datetime.now()
        if self.btn_light["state"] != "disabled":
            self.log_entry('Checked resident: ' + time.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + '.\n', 'full_log.txt')
        self.btn_light["state"] = "disabled"
        self.light_off_thread = threading.Thread(target=self.light_off, daemon=True)
        self.light_off_thread.start()
        
    def light_off(self):
        if self.btn_start["state"] == "disabled" and self.running:
            self.btn_stop["state"] = "normal"
        while True:
            if self.connect_to_network(self.wemo_wifi):
                break
        while True:
            try:
                self.devices = pywemo.discover_devices()
                self.devices[0].off()
                break
            except:
                continue

    def send_email(self, first=False):
        while True:
            if self.connect_to_network(self.ltch_wifi):
                break
            
        while True:
            try:
                # creates SMTP session
                s = smtplib.SMTP('smtp.gmail.com')

                # start TLS for security
                s.starttls()

                # authentication
                s.login(self.from_email, "zdxb nsxv fkir mljf")

                for email in self.to_emails:
                    msg = EmailMessage()
                    msg['Subject'] = 'Vision System Alert: Participant ' + str(self.participant_number)
                    msg['From'] = self.from_email
                    msg['To'] = email
                    msg.set_content('Please check on Participant ' + str(self.participant_number) +
                                    ' as a suspected pain expression has been detected.')
                    s.send_message(msg)

                # terminating the session
                s.quit()
                break
            except:
                continue

    def log_entry(self, entry, filename):
        f = open(os.path.join(self.ssd_location, filename), "a")
        f.write(entry)
        f.close()

    def check_face_angle(self, image):

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = self.face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        y = 360

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                y = angles[1] * 360
        return y

    def run_model(self):
        while self.running:
            with (self.lock):
                if len(self.indices) == 0 or (len(self.indices) > 0 and self.index != self.indices[-1]): 
                    k = self.index
                    self.indices.append(k)
                    try:
                        angle = 0 #self.check_face_angle(self.frame)
                        #print(angle)
                        if -30 <= angle <= 30:
                            pain_score = self.pain_detector.predict_pain(self.frame)
                        else:
                            pain_score = np.nan
                    except:
                        pain_score = np.nan
                    print(pain_score)
                    print()
                    self.pain_scores.append(pain_score)

                    if not self.pain_moment and len(self.pain_scores) >= self.seconds * 3 \
                        and len([p > self.threshold and p is not np.nan for p in itertools.islice(self.pain_scores, len(self.pain_scores)-self.seconds * 3, len(self.pain_scores))]) >= self.high_frames:
                        print(len(self.pain_scores))
                        print(self.seconds * 3)
                        print([p > self.threshold for p in
                                   itertools.islice(self.pain_scores, len(self.pain_scores) - self.seconds * 3,
                                                    len(self.pain_scores))])
                        print(len([p > self.threshold for p in itertools.islice(self.pain_scores, len(self.pain_scores)-self.seconds * 3, len(self.pain_scores))]))
                        print(list(itertools.islice(self.pain_scores, len(self.pain_scores)-self.seconds * 3, len(self.pain_scores))))
                        print(self.high_frames)
                        print()
                        self.pain_moment = True
                        self.start_index = k
                        self.btn_light["state"] = "normal"
                        self.btn_stop["state"] = "disabled"
                        self.text.config(text='Session ongoing: suspected pain expression.')
                        time = datetime.datetime.now()
                        self.log_entry('Suspected pain: ' + time.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + '.\n', 'full_log.txt')
                        self.light_thread = threading.Thread(target=self.turn_on_light, daemon=True)
                        self.light_thread.start()

                    if self.pain_moment and (self.end_index - self.start_index < self.MAX_FRAMES/2 or self.count[-1] - self.count[0] < self.MAX_FRAMES - 1):
                        self.end_index = k

                    elif self.pain_moment and self.end_index - self.start_index >= self.MAX_FRAMES/2 and self.count[-1] - self.count[0] >= self.MAX_FRAMES - 1:
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
        self.btn_select_participant["state"] = "normal"
        self.text.config(text='Please start the session when ready.')
        self.end_time = datetime.datetime.now()
        self.index = 0
        difference = (self.end_time - self.start_time).total_seconds()/60.0
        self.log_entry(',' + self.end_time.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + ',' + str(round(difference, 3)), 'summary_log.txt')
        self.log_entry('Stopped session: ' + self.end_time.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + '.\n', 'full_log.txt')
        self.log_entry('The total duration of this session was ' + str(round(difference, 3)) + ' minutes.\n----------\n', 'full_log.txt')
        self.start_time = None
        self.end_time = None

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
                               'pain_scores_' + str(len(glob.glob1(self.participant, "*.csv")) + 1) + '.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['time', 'index', 'frame', 'pain_score'])
            writer.writerows(zip(times, inds, frms, pain_scores))

    def update_frame(self):
        self.turn_off_light()
        self.btn_select_participant["state"] = "normal"
        self.text.config(text='Please select a participant.')
        while True:
            self.frame = self.vid.read()
            self.frame = cv2.putText(self.frame, str(self.index),(20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                     1, (0, 255, 255),2, cv2.LINE_4)
            if self.running:
                self.frames.append(self.frame)
                self.count.append(self.index)
                self.times.append(datetime.datetime.now().strftime("%Y-%m-%d+%H:%M:%S.%f")[:-3])
                self.index += 1
            
            frame = cv2.resize(self.frame, (int(0.41*self.width), int(0.41*self.height)))
            if self.btn_light["state"] == "normal":
                frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0,0,255))
            else:
                frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(240,240,240))

            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            self.canvas.configure(image=photo)
            self.canvas.image = photo
            self.window.update()

    def select_participant(self): 
        self.reference_images = []
        self.pain_detector.ref_frames = []
        self.participant_label.config(text='')
        for widget in self.photo_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.destroy()
        self.participant = filedialog.askdirectory(initialdir=self.ssd_location)
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
            frame = self.vid.read()
            path = os.path.join(self.participant,
                                'reference_image_' + str(len(glob.glob1(self.participant, "*.jpg")) + 1) + '.jpg')
            try:
                before = len(self.pain_detector.ref_frames)
                self.pain_detector.add_references([np.asarray(frame)])
                after = len(self.pain_detector.ref_frames)
                if before != after:
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
            except:
                pass
            self.window.update()

    def __del__(self):
        self.video_thread.join()
        self.model_thread.join()
        self.email_thread.join()
        self.light_thread.join()
        self.vid.release()
    
    def on_closing(self):
        if self.start_time is not None:
            self.stop_video()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    root.resizable(width=True, height=True)
    model = "model_epoch4.pt"
    threshold = arg_dict.threshold
    seconds = arg_dict.seconds
    high_frames = arg_dict.high_frames
    from_email = arg_dict.from_email
    to_emails = [arg_dict.from_email, arg_dict.to_email]

    # modify the following as needed
    # 0.2422 was chosen based on the optimal threshold based on the ROC AUC curve for FACS pain score of 4
    # we can also use the median values in the table below to make thresholds for each FACS pain score
    ssd = arg_dict.ssd
    ltch_wifi = "PainStudy"
    wemo_wifi = 'WeMo.Switch.' + arg_dict.wemo_code
    # Lumsden: heritagehome@rqhealth.ca
    # CentralHavenSaskatoon: ch.nurses@saskhealthauthority.ca

    location = ssd.split('\\')[-1]
    app = VideoApp(root, location + " Vision System", ssd, model, location, threshold, seconds, high_frames, ltch_wifi, wemo_wifi, from_email, to_emails)
    root.protocol('WM_DELETE_WINDOW', app.on_closing)
    root.mainloop()

# the following table gives the median/mean thresholds for the corresponding FACS pain scores in the Regina Heat Dataset
'''
    Label    median      mean       std   count
0     0.0  0.218241  0.511624  0.684555  192171
1     1.0  0.188460  0.431917  0.525569   19133
2     2.0  0.295027  0.546301  0.588908    4960
3     3.0  0.508348  0.682241  0.574774    2210
4     4.0  0.526920  0.716655  0.550234    1754
5     5.0  0.520900  0.721224  0.580451     922
6     6.0  0.658955  0.802768  0.624160     522
7     7.0  0.747635  0.942009  0.661130     391
8     8.0  0.694499  0.928988  0.641226     422
9     9.0  0.764518  0.871981  0.509774     244
10   10.0  0.788160  0.933165  0.583640     318
11   11.0  1.599810  1.675409  0.708176      68
12   12.0  1.982628  2.016632  0.477232      48
13   13.0  1.658038  1.910605  0.840470      19
14   14.0  2.505320  2.470200  0.356324      59
15   15.0  1.705458  1.747546  0.364064      21
16   16.0  1.405468  1.403963  0.114371      20

Note that 0.2422 is used as the default threshold
since it was was chosen based on the optimal threshold
for the ROC AUC curve for a FACS pain score of 4.
'''
