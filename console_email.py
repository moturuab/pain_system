import smtplib, ssl
from datetime import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tkinter import *
from tkinter import messagebox
from ttkthemes import ThemedTk
from configparser import ConfigParser   # for config/.ini files
from socket import gaierror
import pywemo
from colorama import Fore, Back, Style

config = ConfigParser()
file = 'settings.ini'



class SendEmail:
    # Initializer: #####################################
    def __init__(self, sender_email=None, receiver_email=None, password=None):
        self.port = None    #587 # For starttls
        self.smtp_server = None
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.password = password
        self.flag = False
        self.file_name = "Test" # Default
        self.email_flag = False
        self.light_flag = False
        self.camera_for_algo = True     # Both cameras for the algorithm - Default
        self.DEBUG = False
        self.dev_password = "admin"
        self.time_window = 2            # (seconds) --- Default value   ## Dev
        self.frame_num = 3              # Default value ## Dev
        self.pain_threshold = 0         # Default value ## Dev
        self.key_press = False          # Global variable for the status of keyboard interrupt.

        self.message = MIMEMultipart("alternative")
        self.message["Subject"] = "Patient in pain"
        self.message["From"] = sender_email
        self.message["To"] = receiver_email

        self.html = """\
            <html>
                <body>
                    <p>
                        Hi,<br>
                        Your patient is in pain!<br>
                        <b>Please turn the heat off!</b><br><br>
                        Thanks!<br>
                        --<br>
                        Team Health Psych Lab
                    </p>
                </body>
            </html>
        """

        self.text = """\
            Hi,
            Your patient is in pain!
            Please turn the heat off!
            Thanks!
            --
            Team Health Psych Lab
        """


    # This function actually sends the email: ################################
    def senderFunction(self):
        part1 = MIMEText(self.text, "plain")
        part2 = MIMEText(self.html, "html")

        self.message.attach(part1)
        self.message.attach(part2)

        self.email_flag = True  # signify that email has been sent once.

        context = ssl.create_default_context()
        with smtplib.SMTP(self.smtp_server, self.port) as server:
            server.ehlo() # Can be omitted
            server.starttls(context=context)
            server.ehlo()   # can be omitted
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.receiver_email, self.message.as_string())
            print(Back.LIGHTYELLOW_EX+Fore.BLACK+"Email sent successfully to "+self.sender_email+" from "+self.receiver_email)
            print(Style.RESET_ALL)

    def setLight(self):
        # Light notification:

        self.light_flag = True

    ######################################################################################
    def register_user(self):
        print("Register User")
        config.read(file)  # read settings.ini file

        username_info = username.get()
        # pin_info = account_pin.get()
        to_email_info = to_email.get()
        from_email_info = from_email.get()
        port_info = port_value.get()
        server_info = server_value.get()
        pass_info = password_value.get()


        if len(username_info) == 0 or len(to_email_info) == 0 or len(from_email_info) == 0 or len(
                port_info) == 0 or len(server_info) == 0 or len(pass_info) == 0:                        # making sure all fields are filled
            messagebox.showerror("Error", "All fields required.")
        else:
            if not config.has_section(username_info):  # if username does not exist in .ini file
                # Making sure that port and smtp server entered are correct
                try:
                    server = smtplib.SMTP(server_info, port_info)
                    server.starttls()
                    server.login(to_email_info, pass_info)

                except TimeoutError:
                    messagebox.showerror("Error", "Connection timeout! Make sure you entered the correct port.")
                    return

                except gaierror:
                    messagebox.showerror("Error", "Connection timeout! Make sure you entered the server.")
                    return

                if server.login(to_email_info, pass_info):  # check password for email
                    # updating the config
                    # new_account = 'new'
                    config.add_section(username_info)
                    # config.set(username_info, 'pin', pin_info)
                    config.set(username_info, 'sender_email', to_email_info)
                    config.set(username_info, 'receiver_email', from_email_info)
                    config.set(username_info, 'port', port_info)
                    config.set(username_info, 'smtp_server', server_info)
                    # config.set(username_info, 'password', pass_info)

                    # writing to file
                    with open(file, 'w') as configfile:
                        config.write(configfile)
                        screen1.destroy()
                        messagebox.showinfo("Success", "Credentials verified.")

                else:
                    messagebox.showerror("Error", "Bad credentials")

            else:
                messagebox.showerror("Error", "Username already exists!")

    ######################################################################################
    def register_window(self):
        print("Register")

        # opening a new screen
        global screen1
        screen1 = Toplevel(screen)
        screen1.title("Register")
        screen1.geometry("300x400")

        global username, account_pin, to_email, from_email, port_value, server_value, password_value  # make them accessible and global
        global username_entry, pin_entry, to_email_entry, from_email_entry, port_entry, smtp_entry  # , pass_entry  # make them accessible and global
        username = StringVar()
        # account_pin = StringVar()
        to_email = StringVar()
        from_email = StringVar()
        port_value = StringVar()
        server_value = StringVar()
        password_value = StringVar()

        Label(screen1, text="Please enter the details below:").pack()
        Label(screen1, text="").pack()  # blank-line

        Label(screen1, text="Username").pack()
        username_entry = Entry(screen1, textvariable=username)
        username_entry.pack()

        # Label(screen1, text="PIN").pack()
        # pin_entry = Entry(screen1, show="*", textvariable=account_pin)
        # pin_entry.pack()

        Label(screen1, text="Sender email").pack()
        to_email_entry = Entry(screen1, textvariable=to_email)
        to_email_entry.pack()

        Label(screen1, text="Receiver email").pack()
        from_email_entry = Entry(screen1, textvariable=from_email)
        from_email_entry.pack()

        Label(screen1, text="Port number:").pack()
        port_entry = Entry(screen1, textvariable=port_value)
        port_entry.pack()

        Label(screen1, text="SMTP Server:").pack()
        smtp_entry = Entry(screen1, textvariable=server_value)
        smtp_entry.pack()

        Label(screen1, text="Email password").pack()
        pass_entry = Entry(screen1, show="*", textvariable=password_value)
        pass_entry.pack()

        Label(screen1, text="").pack()  # blank-line
        Button(screen1, text="Register", width=10, height=1, command=self.register_user, relief=GROOVE).pack()

    ######################################################################################
    def onClose(self):
        screen.destroy()  # stops the main loop and interpreter
        print("Session closed.")
        exit()
    ######################################################################################
    def run(self):
        screen_dash.destroy()

    ######################################################################################
    def run1(self):
        screen_run.destroy()



    ######################################################################################
    def login_user(self):
        print("Login")
        #global passw
        global user
        user = username_verify.get()
        self.password = pass_verify.get()
        self.file_name = video_file_name.get()
        self.pain_threshold = pain_threshold.get()
        self.frame_num = frame_num.get()
        self.time_window = time_window.get()
        self.camera_for_algo = algo_cam_verify.get()



        config.read(file)  # read settings.ini file
        # accessing elements: -- config[section][element]
        if len(user) == 0 or len(self.password) == 0 or len(self.file_name)==0 or len(self.pain_threshold)==0 or len(self.frame_num)==0 or len(self.time_window)==0:   # if any fields are empty
            messagebox.showerror("Error", "Please fill all the fields.")
        else:
            if not config.has_section(user):    # if the username is not in any sections of .ini file
                messagebox.showerror("Error", "User does not exist.")

            else:
                # everything is good and working
                self.sender_email = config[user]['sender_email']
                self.receiver_email = config[user]['receiver_email']
                self.port = config[user]['port']
                self.smtp_server = config[user]['smtp_server']

                server = smtplib.SMTP(self.smtp_server, self.port)
                server.starttls()

                if server.login(self.sender_email, self.password):
                    messagebox.showinfo("Success", "Reading and verification was successful!")
                    self.flag = True
                    screen_login.destroy()
                    screen.destroy()
                    global screen_dash
                    screen_dash = Tk()
                    screen_dash.geometry("300x250")
                    screen_dash.title("Dashboard")
                    Label(text="HPL Project", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
                    Label(text="").pack()  # to leave a line in between
                    Button(text="Run", width="30", height="2", command=self.run, relief=GROOVE).pack()

                    screen.mainloop()

                else:
                    messagebox.showerror("Error", "Bad credentials.")

    ######################################################################################
    def login_window(self):
        config.read(file)  # read settings.ini file
        global screen_login
        screen_login = Toplevel(screen)
        screen_login.title("Login")
        screen_login.geometry("300x500")

        global username_login, pass_login, username_verify, pass_verify, video_file_name, file_verify, pain_threshold, pain_verify
        global frame_num, frame_verify, time_window, time_verify, algo_cam_both, algo_cam_verify
        username_login = StringVar()
        pass_login = StringVar()
        file_verify = StringVar()
        pain_verify = StringVar()
        frame_verify = StringVar()
        time_verify = StringVar()
        algo_cam_verify = BooleanVar()

        Label(screen_login, text="Please enter the details below:").pack()
        Label(screen_login, text="").pack()  # blank-line

        Label(screen_login, text="Username:").pack()
        username_verify = Entry(screen_login, textvariable=username_login)
        username_verify.insert(0,"hpl")
        username_verify.pack()

        Label(screen_login, text="Email Password:").pack()
        pass_verify = Entry(screen_login, show="*", textvariable=pass_login)
        pass_verify.insert(0,'Hpl@1234')
        pass_verify.pack()

        Label(screen_login, text="Trial Folder Name:").pack()
        video_file_name = Entry(screen_login, textvariable=file_verify)
        video_file_name.insert(0,"test")
        video_file_name.pack()

        Label(screen_login, text="Minimum Pain Threshold (between 0 and 4):").pack()
        pain_threshold = Entry(screen_login, textvariable=pain_verify)
        pain_threshold.insert(0,"0.2")
        pain_threshold.pack()

        Label(screen_login, text="Window of Time to Detect Pain (seconds):").pack()
        time_window = Entry(screen_login, textvariable=time_verify)
        time_window.insert(0,'3')
        time_window.pack()

        Label(screen_login, text="Number of Frames where Pain Detected:").pack()
        frame_num= Entry(screen_login, textvariable=frame_verify)
        frame_num.insert(0,'5')
        frame_num.pack()

        Label(screen_login, text="Run Algorithm on Both Cameras?").pack()
        Radiobutton(screen_login, variable=algo_cam_verify, value=True, text="Yes").pack()
        Radiobutton(screen_login, variable=algo_cam_verify, value=False, text="No").pack()


        Label(screen_login, text="").pack()  # blank-line
        Button(screen_login, text="Login", width=10, height=1, command=self.login_user, relief=GROOVE).pack()
        # if Button1:
        #     Button1.pack()
        # else:
        #     print("All Field Required!!!!!!!")

    #################################################################################
    def run_button(self):
        global screen_run
        screen_run = Tk()
        screen_run.geometry("300x250")
        screen_run.title("Dashboard")
        Label(text="HPL Project", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
        Label(text="").pack()  # to leave a line in between
        Button(text="Run", width="30", height="2", command=self.run1, relief=GROOVE).pack()

        screen.mainloop()


    #################################################################################
    def dev_login(self):
        print("Dev login")

        dpassw = devpass1.get()
        debug_value = debug_md.get()
        self.camera_for_algo = algo_cam.get()


        if len(dpassw)==0:    # that is, if any fields were empty
            messagebox.showerror("Error", "Please fill all fields.")


        else:       # for empty fields - if they are not empty

            if dpassw == self.dev_password:  # authenticating the dev user
                # Load default values for testing
                self.sender_email = "urpsychlab2020@gmail.com"
                self.receiver_email = "urpsychlab2020@gmail.com"
                self.password = "Hpl@1234"
                self.port = 587
                self.smtp_server = "smtp.gmail.com"

                server = smtplib.SMTP(self.smtp_server, self.port)
                server.starttls()


                self.flag = True
                self.DEBUG = debug_value

                if server.login(self.sender_email, self.password):  # if login was possible for the email
                    # Success:
                    dev_window.destroy()
                    screen.destroy()
                    global screen_run
                    screen_run = Tk()
                    screen_run.geometry("300x250")
                    screen_run.title("Dashboard")
                    Label(text="HPL Project", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
                    Label(text="").pack()  # to leave a line in between
                    Button(text="Run", width="30", height="2", command=self.run1, relief=GROOVE).pack()

                    screen.mainloop()
                else:
                    messagebox.showerror("Login Error", "Could not log in to the email.")

                # dev_window.destroy()
                # screen.destroy()
                #     global screen_dash
                #     screen_dash = Tk()
                #     screen_dash.geometry("300x250")
                #     screen_dash.title("Dashboard")
                #     Label(text="HPL Project", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
                #     Label(text="").pack()  # to leave a line in between
                #     Button(text="Run", width="30", height="2", command=self.run).pack()
                #
                #     screen.mainloop()



            else:
                messagebox.showerror("Error", "Bad credentials.")




    #################################################################################
    def dev_window(self):
        print("Dev window")
        # Dev login:
        global devpass1, frame_number, time_window, tpain, namefile, debug_md, algo_cam
        devpass = StringVar()
        debug_md = BooleanVar()
        algo_cam = BooleanVar()


        global dev_window
        dev_window = Toplevel(screen)
        dev_window.title("Dev Console")
        dev_window.geometry("300x400")

        Label(dev_window, text="Dev Login", bg="#B0CBCB", width="300", height="2", font=("Calibri", 13)).pack()
        Label(dev_window, text="").pack()  # blank-line

        Label(dev_window, text="Debug mode:").pack()
        Radiobutton(dev_window, variable=debug_md, text="True", value=True).pack()
        Radiobutton(dev_window, variable=debug_md, text="False", value=False).pack()


        Label(dev_window, text="Run Algorithm on Both Cameras?").pack()
        Radiobutton(dev_window, variable=algo_cam, value=True, text="Yes").pack()
        Radiobutton(dev_window, variable=algo_cam, value=False, text="No").pack()

        Label(dev_window, text="Password:").pack()
        devpass1 = Entry(dev_window, textvariable=devpass, show="*")
        devpass1.pack()


        Label(dev_window, text="").pack()  # blank-line
        Button(dev_window, text="Go", width=10, height=1, command=self.dev_login, relief=GROOVE).pack()

    #################################################################################
    def main_screen(self):
        global screen  # to make it accessible

        screen = ThemedTk(theme="arc")
        screen.protocol("WM_DELETE_WINDOW", self.onClose)  # handle event when window is closed by user

        screen.geometry("300x250")
        screen.title("HPL Console")
        Label(text="HPL Project", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
        Label(text="").pack()  # to leave a line in between

        Button(text="Login", width="30", height="2", command=self.login_window, relief=GROOVE).pack()
        #Label(text="").pack()  # to leave a line in between

        Button(text="Register", width="30", height="2", command=self.register_window, relief=GROOVE).pack()

        Button(text="Dev", width="30", height="2", command=self.dev_window, relief=GROOVE).pack()
        Label(text="").pack()  # to leave a line in between

        screen.mainloop()
