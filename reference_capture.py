from pain_detector import PainDetector
import cv2
from tkinter import *
from tkinter import messagebox
from ttkthemes import ThemedTk
from tkinter import filedialog
import glob
import os.path
import PySpin
import time
import ntpath
from PIL import Image


###########
pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt')
###########

class ReferenceCapture:

    # Initializer: #####################################
    def __init__(self):
        self.NUM_IMAGES = 1
        self.destination_folder = ''
        self.both_cameras = True
        self.filenames = []


    ###########################################################################################################
    def acquire_images(self, cam, nodemap, nodemap_tldevice):
        """
        This function acquires and saves 10 images from a device.

        :param cam: Camera to acquire images from.
        :param nodemap: Device nodemap.
        :param nodemap_tldevice: Transport layer device nodemap.
        :type cam: CameraPtr
        :type nodemap: INodeMap
        :type nodemap_tldevice: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """

        print('*** IMAGE ACQUISITION ***\n')
        try:
            result = True

            # Set acquisition mode to continuous
            #
            #  *** NOTES ***
            #  Because the example acquires and saves 10 images, setting acquisition
            #  mode to continuous lets the example finish. If set to single frame
            #  or multiframe (at a lower number of images), the example would just
            #  hang. This would happen because the example has been written to
            #  acquire 10 images while the camera would have been programmed to
            #  retrieve less than that.
            #
            #  Setting the value of an enumeration node is slightly more complicated
            #  than other node types. Two nodes must be retrieved: first, the
            #  enumeration node is retrieved from the nodemap; and second, the entry
            #  node is retrieved from the enumeration node. The integer value of the
            #  entry node is then set as the new value of the enumeration node.
            #
            #  Notice that both the enumeration and the entry nodes are checked for
            #  availability and readability/writability. Enumeration nodes are
            #  generally readable and writable whereas their entry nodes are only
            #  ever readable.
            #
            #  Retrieve enumeration node from nodemap

            # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images. Because the example calls for the
            #  retrieval of 10 images, continuous mode has been set.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            # if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            #     device_serial_number = node_device_serial_number.GetValue()
            #     print('Device serial number retrieved as %s...' % device_serial_number)

            # Retrieve, convert, and save images
            for i in range(self.NUM_IMAGES):
                try:

                    #  Retrieve next received image
                    #
                    #  *** NOTES ***
                    #  Capturing an image houses images on the camera buffer. Trying
                    #  to capture an image that does not exist will hang the camera.
                    #
                    #  *** LATER ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.
                    image_result = cam.GetNextImage()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    #  Ensure image completion
                    #
                    #  *** NOTES ***
                    #  Images can easily be checked for completion. This should be
                    #  done whenever a complete image is expected or required.
                    #  Further, check image status for a little more insight into
                    #  why an image is incomplete.
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:

                        #  Print image information; height and width recorded in pixels
                        #
                        #  *** NOTES ***
                        #  Images have quite a bit of available metadata including
                        #  things such as CRC, image status, and offset values, to
                        #  name a few.
                        width = image_result.GetWidth()
                        height = image_result.GetHeight()
                        print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                        #  Convert image to mono 8
                        #
                        #  *** NOTES ***
                        #  Images can be converted between pixel formats by using
                        #  the appropriate enumeration value. Unlike the original
                        #  image, the converted one does not need to be released as
                        #  it does not affect the camera buffer.
                        #
                        #  When converting images, color processing algorithm is an
                        #  optional parameter.
                        # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                        # Create a unique filename - timestamp.
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        filename = os.path.normpath(self.destination_folder.get()+'//Cam'+str(camera_index)+'_'+timestr+'_(%d).jpg' % (i))

                        #  Save image
                        #
                        #  *** NOTES ***
                        #  The standard practice of the examples is to use device
                        #  serial numbers to keep images of one device from
                        #  overwriting those of another.
                        #  image_converted.Save(filename) <-original
                        image_result.Save(filename)
                        print('Image saved at %s' % filename)

                        #  Release image
                        #
                        #  *** NOTES ***
                        #  Images retrieved directly from the camera (i.e. non-converted
                        #  images) need to be released in order to keep from filling the
                        #  buffer.
                        image_result.Release()
                        print('')

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            #  End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
            cam.EndAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    ###########################################################################################################
    def run_single_camera(self, cam):
        try:
            result = True

            # Retrieve TL device nodemap and print device information
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            # Acquire images
            result &= self.acquire_images(cam, nodemap, nodemap_tldevice)  ## Function call !!!

            # Deinitialize camera
            cam.DeInit()


        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False


        return result

    ###########################################################################################################

    ###########################################################################################################
    # This function will capture reference images.
    def capture_window(self):
        # Lets the user select the folder for reference images to be saved.
        self.destination_folder = StringVar()
        folder_path = filedialog.askdirectory()
        self.destination_folder.set(folder_path)
        self.both_cameras = BooleanVar()
        #print(both_cam_verify.get())
        self.both_cameras = both_cam_verify.get()

        global camera_index
        camera_index = 0

        self.NUM_IMAGES = int(number.get())

        result = True
        try:
            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            # Get current library version
            version = system.GetLibraryVersion()
            # Retrieve list of cameras from the system
            cam_list = system.GetCameras()

            num_cameras = cam_list.GetSize()
            print('Number of cameras detected: %d' % num_cameras)

            # Finish if there are no cameras
            if num_cameras == 0:
                # Clear camera list before releasing system
                cam_list.Clear()

                # Release system instance
                system.ReleaseInstance()

                print('Not enough cameras!')
                return False

            # Run example on each camera
            for i, cam in enumerate(cam_list):
                if self.both_cameras==False and i==1:
                    break
                else:
                    result &= self.run_single_camera(cam)  ## function call !!!
                    camera_index = camera_index + 1

            messagebox.showinfo("OK", "Images saved at \n" +str(os.path.normpath(self.destination_folder.get())))
            # ---------------------------------
            # Release reference to camera
            # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
            # cleaned up when going out of scope.
            # The usage of del is preferred to assigning the variable to None.
            del cam

            # Clear camera list before releasing system
            cam_list.Clear()

            # Release system instance
            system.ReleaseInstance()

        except:
            print("Couldn't capture images.")

        return result

    ###########################################################################################################
    def insertfiles(self):
        #os.chdir(folder.get())      # setting current directory path to reference image path

        count = 0

        for image in glob.glob(folder.get() + "\\*png"):
            lst.insert(END, ntpath.basename(image))
            self.filenames.append(os.path.normpath(image))

        # for filename in glob.glob(folder.get() + "\\*png"):
        #     #new_filename = ntpath.basename(filename)        ## To take out directory structure for a filename
        #     lst.insert(END, ntpath.basename(filename))
            count = count + 1

        return count


    ###########################################################################################################
    def show(self, event):

        # bi = Image.open(self.filenames[int(lst.curselection()[0])])
        # bi.show()

        img = PhotoImage(file=self.filenames[int(lst.curselection()[0])])
        # w, h = img.width(), img.height()
        canvas.image = img
        # canvas.config(width=w, height=h)
        canvas.create_image(100, 100, image=img, anchor=NW)

        # n = lst.curselection()  # Return the indices of currently selected item.
        # try:
        #     filename = lst.get(n)  # Get list of items from FIRST to LAST (included)
        #     # print("-----------"+filename)
        #     if any(filename):
        #         img = PhotoImage(file=filename)
        #         w, h = img.width(), img.height()
        #         canvas.image = img
        #         canvas.config(width=w, height=h)
        #         canvas.create_image(0, 0, image=img, anchor=NW)
        #     else:
        #         messagebox.showerror("Error", "No PNG files to display.")
        #
        # except TclError:
        #     display_screen.destroy()
        #     messagebox.showerror("Error", "Nothing to show.")

    ###########################################################################################################
    def display_window(self):       # to display file names and images in the same window. Interacts with insertfiles() and show() simultaneously.
        global display_screen, lst, canvas, folder, lab
        folder = StringVar()
        folder_dir = filedialog.askdirectory()
        folder.set(folder_dir)

        self.destination_folder = StringVar()
        self.destination_folder.set(folder_dir)

        display_screen = Toplevel(ref_screen)
        display_screen.geometry("800x400+300+50")

        lst = Listbox(display_screen, width=40)

        counter = self.insertfiles()        # Number of PNG files in folder

        if counter!=0:      # if some PNG files found, display them
            lst.pack(side="left", fill=BOTH, expand=0)

            lst.bind("<<ListboxSelect>>", self.show)
            canvas = Canvas(display_screen, width = 300, height = 200, bg = 'grey', relief=RIDGE)
            #canvas = Canvas(display_screen, relief=GROOVE)
            canvas.pack(expand=YES, fill=BOTH)

        else:               # otherwise, show an error message.
            display_screen.destroy()
            messagebox.showerror("", "No PNG files found in the folder:\n"+str(os.path.normpath(folder.get())))


    ###########################################################################################################
    def verify_window(self):
        # self.destination_folder = StringVar()
        #print(self.destination_folder)
        #print(self.destination_folder.get())

        if self.destination_folder == '':       # no reference images have been captured previously - asks to set the destination folder
            Msg = messagebox.askquestion("Question", "Proceed to select the folder for reference images?")
            if Msg=="yes":  # pick directory
                folder_ref = StringVar()
                fpath = filedialog.askdirectory()
                folder_ref.set(fpath)

                self.destination_folder = StringVar()
                self.destination_folder.set(fpath)

                #os.chdir(folder_ref.get())

                print("Verifying reference images...")

                for ref_image in glob.glob(self.destination_folder.get() + '\\*[j][p][g]'):
                    ref_frame = cv2.imread(ref_image)
                    # Sending image through verify_reference()
                    try:
                        new_image = pain_detector.verify_refenerece_image(ref_frame)
                        i = 0
                        new_name1 = os.path.splitext(os.path.normpath(ref_image))[0]
                        cv2.imwrite(new_name1+".png", new_image)
                        i = i+1
                    except TypeError:
                        print("No faces were detected for image " + str(os.path.normpath(ref_image)))


            else:
                messagebox.showerror("Error", "Reference folder not set.")

        else:
            print("Verifying reference images...")
            print(os.path.normpath(self.destination_folder.get()))

            for ref_image in glob.glob(self.destination_folder.get()+'/*[j][p][g]'):
                ref_frame = cv2.imread(ref_image)
                #Sending image through verify_reference()
                try:
                    new_image = pain_detector.verify_refenerece_image(ref_frame)
                    new_name = os.path.splitext(os.path.normpath(ref_image))[0]   # Extracting the name of the JPG image file by removing the file extension.
                    cv2.imwrite(new_name+".png", new_image)
                except TypeError:
                    print("No faces were detected for image "+str(os.path.normpath(ref_image)))

        messagebox.showinfo("", "Complete.")

    ###########################################################################################################
    def onClose(self):
        ref_screen.destroy()  # stops the main loop and interpreter
        #print("References taken")
    ###########################################################################################################
    # Menu for window for capturing reference images.
    def ref_menu(self):

        global ref_screen, number, both_cam_verify


        ref_screen = ThemedTk(theme="arc")
        ref_screen.protocol("WM_DELETE_WINDOW", self.onClose)  # handle event when window is closed by user

        ref_screen.geometry("500x500")
        ref_screen.title("Reference images")

        Label(text="Capture Settings", bg="#7c797d", width="300", height="2", font=("Calibri", 13)).pack()
        Label(text="").pack()  # to leave a line in between
        Label(ref_screen, text="Bursts for each camera:").pack()
        number = Spinbox(ref_screen, from_=1, to=10)
        number.pack()

        Label(text="").pack()

        both_cam_verify = BooleanVar()

        Label(ref_screen, text="Capture references for all cameras?").pack()
        Radiobutton(ref_screen, variable=both_cam_verify, value=True, text="Yes").pack()
        Radiobutton(ref_screen, variable=both_cam_verify, value=False, text="No").pack()

        Label(text="").pack()

        Label(text="Reference Image Capture", bg="#9379F3", width="300", height="2", font=("Calibri", 13)).pack()
        Label(text="").pack()  # to leave a line in between

        Button(text="Take reference picture(s)", width="30", height="2", command=self.capture_window, relief=GROOVE).pack()
        # Label(text="").pack()  # to leave a line in between

        Button(text="Verify reference picture(s)", width="30", height="2", command=self.verify_window, relief=GROOVE).pack()
        # Label(text="").pack()  # to leave a line in between

        Button(text="Browse images", width="30", height="2", command=self.display_window, relief=GROOVE).pack()

        #ref_screen.after(1000, ref_screen.destroy)
        ref_screen.mainloop()
