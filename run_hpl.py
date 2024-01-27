# Imports:
##
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import freeze_support
import re
from pain_detector import PainDetector  # to be kept on top
from console_email import SendEmail  # to be kept on top
from reference_capture import ReferenceCapture
from concurrent.futures import ProcessPoolExecutor
import threading
import logging
import multiprocessing

import os
import PySpin
from PySpin import Image
from threading import Thread, Event
from queue import Queue
import time
from glob import glob
import cv2
import csv

import \
    pywemo  # pywemo repository from github link : http://github.com/pavoni/pywemo, minor changes done to avoid errors
from datetime import datetime
from pytz import timezone
from colorama import init, deinit
from colorama import Fore, Back, Style
from skimage import io

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

QUEUE_AVI = True
#####################################################################################################################
# Global variables:
##
consoleEmail = SendEmail()
reference_verify = ReferenceCapture()


# consoleEmail.main_screen()
###
class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2


###

chosenAviType = AviType.H264  # change me!
NUM_IMAGES = 500  # number of images to grab
target_images_folder = 'target_images'  # folder to save

devices = pywemo.discover_devices()  # takes computation time
pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt')


#####################################################################################################################
# Methods:
###
# Prepare the video writers
def get_video_writers(path_to_file, nodemap, nodemap_tldevice, width, height):
    """
        This function prepares, saves, and cleans up an AVI video from a vector of images.

        :param nodemap: Device nodemap.
        :param nodemap_tldevice: Transport layer device nodemap.
        :param image: List of images to save to an AVI video.
        :type nodemap: INodeMap
        :type nodemap_tldevice: INodeMap
        :type images: list of ImagePtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
    if consoleEmail.DEBUG:
        print('*** CREATING VIDEO ***')

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
            device_serial_number = node_serial.GetValue()
            if consoleEmail.DEBUG:
                print('Device serial number retrieved as %s...' % device_serial_number)

        # Get the current frame rate; acquisition frame rate recorded in hertz

        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
            if consoleEmail.DEBUG:
                print('Unable to retrieve frame rate. Aborting...')
            return False
        # node_acquisition_framerate

        framerate_to_set = node_acquisition_framerate.GetValue()
        print(f"Frame rate for creating the video file is : {framerate_to_set}")

        if consoleEmail.DEBUG:
            print('Frame rate to be set to %d...' % framerate_to_set)  # framerate_to_set #consoleEmail.frame_rate

        # Select option and open AVI filetype with unique filename
        avi_recorder = PySpin.SpinVideo()

        if chosenAviType == AviType.UNCOMPRESSED:
            avi_filename = str(consoleEmail.file_name) + '-Uncompressed-%s' % device_serial_number

            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set

        elif chosenAviType == AviType.MJPG:
            avi_filename = str(consoleEmail.file_name) + '-MJPG-%s' % device_serial_number

            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 30

        elif chosenAviType == AviType.H264:
            avi_filename = str(consoleEmail.file_name) + '-H264-%s' % device_serial_number

            option = PySpin.H264Option()
            option.frameRate = framerate_to_set
            option.bitrate = 5000000  # decides the level of compression
            option.height = height  # image[0].GetHeight()
            if consoleEmail.DEBUG:
                print('image height = %d' % (option.height))
            option.width = width  # image[0].GetWidth()
            if consoleEmail.DEBUG:
                print('image height = %d' % (option.width))

        else:
            print('Error: Unknown AviType. Aborting...')
            return False
        avi_filename = path_to_file + "/" + avi_filename
        avi_recorder.Open(avi_filename, option)
    except Exception as e:
        print(e)

    return avi_recorder


# Method to save image list to video (avi)
def save_list_to_avi(nodemap, nodemap_tldevice, image, width, height):
    """
    This function prepares, saves, and cleans up an AVI video from a vector of images.

    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param image: List of images to save to an AVI video.
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type images: list of ImagePtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    if consoleEmail.DEBUG:
        print('*** CREATING VIDEO ***')

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
            device_serial_number = node_serial.GetValue()
            if consoleEmail.DEBUG:
                print('Device serial number retrieved as %s...' % device_serial_number)

        # Get the current frame rate; acquisition frame rate recorded in hertz

        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
            if consoleEmail.DEBUG:
                print('Unable to retrieve frame rate. Aborting...')
            return False
        # node_acquisition_framerate

        framerate_to_set = node_acquisition_framerate.GetValue()

        if consoleEmail.DEBUG:
            print('Frame rate to be set to %d...' % framerate_to_set)  # framerate_to_set #consoleEmail.frame_rate

        # Select option and open AVI filetype with unique filename
        avi_recorder = PySpin.SpinVideo()

        if chosenAviType == AviType.UNCOMPRESSED:
            avi_filename = str(consoleEmail.file_name) + '-Uncompressed-%s' % device_serial_number

            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set

        elif chosenAviType == AviType.MJPG:
            avi_filename = str(consoleEmail.file_name) + '-MJPG-%s' % device_serial_number

            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 30

        elif chosenAviType == AviType.H264:
            avi_filename = str(consoleEmail.file_name) + '-H264-%s' % device_serial_number

            option = PySpin.H264Option()
            option.frameRate = framerate_to_set
            option.bitrate = 5000000  # decides the level of compression
            option.height = height  # image[0].GetHeight()
            if consoleEmail.DEBUG:
                print('image height = %d' % (option.height))
            option.width = width  # image[0].GetWidth()
            if consoleEmail.DEBUG:
                print('image height = %d' % (option.width))

        else:
            print('Error: Unknown AviType. Aborting...')
            return False

        avi_recorder.Open(avi_filename, option)

        # Construct and save AVI video

        if consoleEmail.DEBUG:
            print('Appending %d images to AVI file: %s.avi...' % (len(image), avi_filename))

        # for i in range(len(image)):
        #     image[i].save('captured/photo' + i + '.jpg', 'JPEG')
        #
        #     save_img(image[i])
        #     #
        # export_images = image[i]
        #
        # ref_image_filename = 'RefImage_'  % i % '_' % device_serial_number
        #
        # Image.save(ref_image_filename % '.jpg')

        # print("Length of images is %d" %(len(image)))

        for i in range(len(image)):
            avi_recorder.Append(image[i])
            # print('Appended image %d...' % i)

        # Close AVI file
        avi_recorder.Close()
        print('Video saved at %s.avi' % avi_filename)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


#####################################################################################################################
# Method to acquire image for the thread from cameras:


def save_images(image, path):
    try:
        pass
        # cv2.imwrite(path,image)
        image.Save(path)
        # print(path + " saved in the disc")
        # image.Release()
    except Exception as e:
        print(e)


def acquire_image(cam, queue_temp, images_queue, event_for_main, i, executor, queue_avi):
    # print("In the thread")
    n = 0;  # indicating the image number
    # flag = True
    # while True:

    # submit tasks to generate files

    while not consoleEmail.key_press:
        # fetching one value from the queue, the thread will wait for a value in the queue
        q = queue_temp.get()

        if q is not None:
            start_ac = time.time()
            try:
                result = True
                # acquire image
                n += 1  # incrementing the image counter

                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    logging.debug('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
                    print('Camera %d Image incomplete with image status %d ... \n' % (
                        i, image_result.GetImageStatus()))
                else:
                    # Print image information
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    # logging.debug('Camera %d grabbed image %d, width = %d, height = %d' % (i, n, width, height))
                    if consoleEmail.DEBUG:
                        print('Camera %d grabbed image %d, width = %d, height = %d' % (i, n, width, height))

                    # Convert image to BayerRG8
                    tmp = image_result.Convert(PySpin.PixelFormat_BayerRG8, PySpin.HQ_LINEAR)
                    # images_temp.append(tmp)
                    if queue_avi:
                        images_queue.put(tmp)
                    ##indidicating the main thread that

                    # images.append(image_result)
                    # if n == 0:
                    if consoleEmail.DEBUG:
                        print("Saving image")

                    path = target_images_folder + "\\Cam " + str(i) + "\\Image %d.jpg" % (
                        n)  # todo: save files separately for each camera
                    _ = executor.submit(save_images, tmp, path)
                    # save_images(image_result,target_images_folder + "\\Cam " + str(i) + "\\Image %d.jpg" % (n))
                    event_for_main.wait()
                    if consoleEmail.DEBUG:
                        print("\n Event true for cam %d" % i)

                # Release image
                image_result.Release()
                # print()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False
            # try:
            #     time_past = time.time() - start_ac
            #     logging.debug(f'Cam {i} Frame rate is: {1 / (time_past + .0000001)} ')
            # except:
            #     pass
        else:
            # print("Finished frame acquisition for camera %d  \n" % i)
            break;

    return result


#####################################################################################################################
def print_device_info(nodemap, cam_num):
    """
    This function prints the device information of the camera from the transport
    layer;
    :param nodemap: Transport layer device nodemap.
    :param cam_num: Camera number.
    :type nodemap: INodeMap
    :type cam_num: int
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    if consoleEmail.DEBUG:
        print('Printing device information for camera %d... \n' % cam_num)

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:  # Printing device information.
                node_feature = PySpin.CValuePtr(feature)
                # print('%s: %s' % (node_feature.GetName(),
                #                   node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')
        print()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


#####################################################################################################################
def light():
    try:
        # devices = pywemo.discover_devices()  # takes computation time
        devices[0].off()  # Turning the device off it is on , just to make sure when On
        # command is executed there is a change in state,
        # in the above statement index is specified 0 as a ideal situation was taken where-in there is only one wemo device
        # time.sleep(2)
        devices[0].on()  # Turning device on

    except IndexError:
        print("!!! WeMo device not connected.")


#####################################################################################################################
# Pain detection method, referencing the pain detection algorithm and models.
def detect_pain(i, fps, csv_file):

    global light_flag, pain_frames, timer
    timer = {}
    pain_frames = {}
    light_flag = False

    time.sleep(2 / 30)  # ToDo Change back to 2/30
    original_size = 0
    pain_frames[i] = 0

    print("-------------")
    print(Back.LIGHTCYAN_EX + Fore.BLACK + "Detecting pain for Camera " + str(i))
    # print(Style.RESET_ALL)

    # start_time = datetime.now(timezone("America/Regina"))
    start_time = time.time()
    if consoleEmail.DEBUG:
        print(start_time)

    # window_time = datetime.now(timezone("America/Regina"))
    timer[i] = time.perf_counter()
    time_frame = {}
    index = 0
    status = ""
    # counter = 0

    while not consoleEmail.key_press:  # while no Keyboard Interrupt is encountered.

        size = len([name for name in os.listdir(target_images_folder + "/Cam " + str(i) + "/") if
                    os.path.isfile(os.path.join(target_images_folder + "/Cam " + str(i), name))])
        size = size - 1  # since the last frame is not still completely writen.

        # print("Image: " + str(os.path.isfile(target_images_folder + "\\Cam " + str(i) + "\\Image %d.jpg" % (
        #     size))))  # Prints true or false. Answers the question: Does the file exist?
        # time.sleep(1/60)

        if size > original_size:  ## to prevent an endless loop

            image_name = "/Cam " + str(i) + "/Image " + str(
                size) + ".jpg"  # todo: to keep this line for both cameras' information to be saved to .csv
            # image_name = "Cam_number_0_image_" + str(size) + ".jpg"
            filename = target_images_folder + image_name
            file_created_time = datetime.fromtimestamp(os.path.getctime(filename))
            target_frame = cv2.imread(target_images_folder + image_name)
            # print (target_images_folder + image_name)
            # print(target_frame.shape)
            # with OpenCV
            # print(target_images_folder+image_name)
            # try:
            #     target_frame = io.imread(target_images_folder + image_name)         # with SkImage
            # except OSError:
            #     continue

            # target_frame = cv2.imread("results/test/captured_images/Image 2630.jpg")  # only for testing purposes. todo: remove this comment
            # print("ok")
            # print(consoleEmail.key_press)
            # if pain_detector.face_detector(target_frame) != None:      #todo: remove, if exceptions work.
            try:

                pain_estimate = pain_detector.predict_pain(target_frame)

                video_tstmp = size / fps  # seconds

                logging.debug("Threads:")
                logging.debug(threading.active_count())
                # img_tstmp = os.path.getctime(target_images_folder + "\\Cam "+str(i)+"\\Image %d.jpg" % (size))

                # writing to a .csv

                with open(csv_file, mode="a") as records:
                    output = csv.writer(records, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)


                    # output.writerow([datetime.now(timezone("America/Regina")), image_name, pain_estimate, consoleEmail.pain_threshold, consoleEmail.frame_num, consoleEmail.time_window, ms_time])
                    output.writerow([video_tstmp, datetime.now(timezone("America/Regina")),str(file_created_time),str(file_created_time), image_name, pain_estimate,
                                     consoleEmail.pain_threshold, consoleEmail.frame_num, consoleEmail.time_window
                                     ])

                # window_time[i] = time.time()    # for each thread
                # pain_estimate = 1
                if pain_estimate > float(consoleEmail.pain_threshold):  # should be > than value input by user
                    # start timer
                    status = "In pain"
                    index = pain_frames[i]
                    time_frame[index] = float('%.2f' % time.perf_counter())

                    # then in this frame patient is in pain:
                    if consoleEmail.DEBUG == False:
                        print(Fore.RED + status + "\t" + image_name)
                        logging.debug(status + "\t" + image_name)
                        # print(Style.RESET_ALL)

                    # add one to counter - signal when counter == frame_num:
                    pain_frames[i] = pain_frames[i] + 1
                    if consoleEmail.DEBUG:
                        print("Pain frames:")
                        print(pain_frames[i])
                        print("\n")

                    if pain_frames[i] >= float(consoleEmail.frame_num) and index >= int(consoleEmail.frame_num) - 1:
                        difference = time_frame[index] - time_frame[index - (int(consoleEmail.frame_num) - 1)]

                        if difference <= float(consoleEmail.time_window):
                            # Send email for pain
                            if consoleEmail.email_flag == False:  # send email and activate light and set email_flag to True
                                consoleEmail.senderFunction()
                            # Set light notification, only once.
                            if consoleEmail.light_flag == False:
                                consoleEmail.setLight()
                                light()


                else:
                    status = "No pain"
                    if consoleEmail.DEBUG == False:
                        print(status)

                        # print(image_name)

                continue


            except Exception as e:
                # else:
                # print(traceback.format_exc())
                # print(e)
                # print("Exception.")
                continue

        elif size == original_size and size != 0:
            break
        print("ok")
        original_size = size
        # -----------END OF WHILE--------------------------------------------------------------
    logging.debug("Pain Detection is Done. ")
    return True


# write files inside the queue untile reach a None object which means the end.
def create_avi_from_queue(avi_writer, images_queue):
    while True:
        start = time.time()
        img = images_queue.get()
        logging.debug("writing a new frame")
        if img is not None:
            avi_writer.Append(img)
            end = time.time() - start
            logging.debug("Write an frame and the speed is : {} (fps)".format(1 / end))
            logging.debug(f"The Queue size is : {images_queue.qsize()} ")
        else:
            avi_writer.Close()
            print("Video Saved....")
            return


#####################################################################################################################
def run_multiple_cameras(cam_list):
    """
    :param cam_list: List of cameras
    :type cam_list: CameraList
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    images = {}
    # images_copy_for_pain = {}
    queues = {}
    event_for_main = {}
    try:
        result = True
        # Retrieve transport layer nodemaps and print device information for
        # each camera
        if consoleEmail.DEBUG:
            print('*** DEVICE INFORMATION ***\n')

        for i, cam in enumerate(cam_list):
            # Retrieve TL device nodemap
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Print device information
            result &= print_device_info(nodemap_tldevice, i)

            # Adding a image list for each camera
            images[i] = list()
            # Adding a queue for each camera
            queues[i] = Queue()

            # Adding an event for each thread
            event_for_main[i] = Event()

        # Initialize each camera
        # *** LATER ***
        # Each camera needs to be deinitialized once all images have been
        # acquired.
        for i, cam in enumerate(cam_list):

            # Initialize camera
            cam.Init()

            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                return False

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                Aborting... \n' % i)
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            if consoleEmail.DEBUG:
                print('Camera %d acquisition mode set to continuous...' % i)

            # Begin acquiring images
            cam.BeginAcquisition()

            if consoleEmail.DEBUG:
                print('Camera %d started acquiring images...' % i)

            print()

        # Retrieve, convert, and save images for each camera
        t = {}
        barrier = threading.Barrier(len(cam_list) + 1)
        for i, cam in enumerate(cam_list):
            # Acquire image on all cameras
            # t[i]=Thread(target = acquire_image,args=(cam_list[i],queues[i],images[i],event_for_main[i],i))
            t[i] = multiprocessing.Process(target=acquire_image, args=(cam_list[i], queues[i], images[i], barrier, i))
            logging.debug("Acquire image --- count:")
            logging.debug(threading.active_count())
            t[i].start()

        # thread for Pain Detection
        pain_thread = {}
        for i, cam in enumerate(cam_list):
            if consoleEmail.camera_for_algo == False:  ## Checking for consoleEmail.camera_for_algo: if True, then both cameras needed, If False, only one camera. -- only for pain_thread
                if i == 1:
                    break
            # pain_thread[i] = Thread(target = detect_pain, args=(images[i], i))
            # pain_thread[i] = multiprocessing.Process(target=detect_pain, args=(images[i], i))
            logging.debug("PAIN --- count:")
            logging.debug(threading.active_count())
            # pain_thread[i].start()

        while (True):
            try:
                # time.sleep(1/30)   #30 frames per second # int(consoleEmail.frame_rate) for taking 1 picture per frame_rate ie frame_rate = 30 means 1 picture every 1/30 seconds

                for i, cam in enumerate(cam_list):
                    # Adding 1 for every frame to be captured
                    queues[i].put("1")
                    # print("\nAdding 1 to the queue for cam %d" % i)

                # for i, cam in enumerate(cam_list):
                #     # Waiting for each thread to signal successful frame capture
                #     event_for_main[i].wait()
                barrier.wait()

            except KeyboardInterrupt:
                consoleEmail.key_press = True
                break

        # Ending the thread by adding a None in the queue
        for i, cam in enumerate(cam_list):
            queues[i].put(None)

        # making sure the video generation by frame compilation only starts after all the frames are captured

        for i, cam in enumerate(cam_list):
            if consoleEmail.camera_for_algo == False:  ## Checking for consoleEmail.camera_for_algo: if True, then both cameras needed, If False, only one camera. -- only for pain_thread
                if i == 1:
                    break
            t[i].join()
            pain_thread[i].join()

        # Deinitialize each camera
        for i, cam in enumerate(cam_list):
            # End acquisition

            cam.EndAcquisition()

            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()
            # result &= save_img(images[i])

            result &= save_list_to_avi(nodemap, nodemap_tldevice, images[i])

            # Deinitialize camera
            cam.DeInit()

            # Release reference to camera
            del cam

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


#################################################################################################################


def create_directory(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
    except Exception as e:
        return False
    return True


def create_directories(trial_name):
    directories = {}
    result_dir = "results"
    directories["root"] = result_dir
    try:
        # create the results folder
        if not create_directory(result_dir):
            print(f"Something went wrong with creating the director:  {result_dir}")

        trial_dir = result_dir + "/" + trial_name
        directories["trial"] = trial_dir
        if not create_directory(trial_dir):
            print(f"Something went wrong with creating the director:  {trial_dir}")

        ref_dir = trial_dir + "/" + "reference_images"
        directories["ref"] = ref_dir
        if not create_directory(ref_dir):
            print(f"Something went wrong with creating the director:  {trial_dir}")

        cap_dir = trial_dir + "/captured_images"
        directories["images"] = cap_dir
        if not create_directory(cap_dir):
            print(f"Something went wrong with creating the director:  {trial_dir}")
            # ,,,,,,
            # create the records.csv file here

        record_csv = trial_dir + "/" + 'records.csv'
        directories["csv"] = record_csv
        with open(record_csv, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(
                ["Seconds", "Output produced at (Pain algo timestamp)", "capturing time", "Image_Name", "Pain_Value", "Pain_Threshold",
                 "Frames per Window", "Analysis Window (s)"])

        print(f"All the direcotries and files are created for the trial: {trial_name}")
    except Exception as e:
        print(f"Error in making the directories: {e}")
    return directories


#####################################################################################################################
def main(queue_avi=False):
    """
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    consoleEmail.main_screen()

    print(f"Creating a folder for: {consoleEmail.file_name}.")
    directories = create_directories(consoleEmail.file_name)

    executor = ThreadPoolExecutor(6)

    global devices, target_images_folder
    devices = pywemo.discover_devices()

    print('Device: ', pain_detector.device)

    init(autoreset=True)  # For colored output on the terminal.

    try:
        test_file = open('test.txt', 'w+')
        logging.basicConfig(filename=f"{directories['trial']}/log.txt", level=logging.DEBUG)

        # Check if reference image folder exists, if not create one:
        target_images_folder = directories["images"]
        if not os.path.exists(target_images_folder):
            os.mkdir(target_images_folder)



    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Function to capture reference images

    print("Capturing reference images...")
    ref_folder = ''
    reference_verify.ref_menu()

    if reference_verify.destination_folder == '':
        ref_folder = directories["ref"]  # default folder for reference images, if nothing has been selected.
    else:
        ref_folder = reference_verify.destination_folder.get()

    # Reference images for the algorithm
    try:
        ref_frame_list = []
        for ref_image in glob(ref_folder + '/*[j][p][g]'):
            # print(reference_verify.destination_folder.get())
            # for ref_image in glob(reference_verify.destination_folder.get()+'/*[j][p][g]'):
            ref_frame = cv2.imread(ref_image)
            ref_frame_list.append(ref_frame)

        if ref_frame_list:
            pain_detector.add_references(ref_frame_list)
        else:
            print("No images in the reference_images folder.")
            input('Press Enter to exit...')

            return True

    except Exception as e:
        print(e)
        print("Reference image error!")
        input('Press Enter to exit...')
        return True

    # Email set-up
    if consoleEmail.DEBUG:
        print("SETUP for emails")
    # global sender_email, receiver_email, password
    # sender_email = input("Please enter your/sender's email address: ")
    # receiver_email = input("Please enter the receiver's email address: ")
    # password = getpass()

    # global consoleEmail
    # consoleEmail = SendEmail()

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    # print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    # print(cam_list)

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Create folders inside target_image_folder for each camera:
    print("Creating folder for each camera inside target_images...")
    for i in range(num_cameras):
        if not os.path.exists(target_images_folder + "/Cam " + str(i)):  # if the folder does not exist, create one.
            os.mkdir(target_images_folder + "/Cam " + str(i))

    # Run on all cameras
    if consoleEmail.DEBUG:
        print('Running for all cameras...')

    # result = run_multiple_cameras(cam_list)

    ####################################################################
    ####################################################################
    # Adding run_multiple_cameras code here because multiprocessing to be in main():
    images_queues = {}
    # images_copy_for_pain = {}
    queues = {}
    event_for_main = {}

    result = True

    # Retrieve transport layer nodemaps and print device information for
    # each camera
    if consoleEmail.DEBUG:
        print('*** DEVICE INFORMATION ***\n')

    for i, cam in enumerate(cam_list):
        # Retrieve TL device nodemap
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Print device information
        result &= print_device_info(nodemap_tldevice, i)

        # Adding a image list for each camera
        images_queues[i] = Queue()
        # Adding a queue for each camera
        queues[i] = Queue()

        # Adding an event for each thread
        event_for_main[i] = Event()

    # Initialize each camera
    # *** LATER ***
    # Each camera needs to be deinitialized once all images have been
    # acquired.
    cam_fps = 0
    for i, cam in enumerate(cam_list):

        # Initialize camera
        cam.Init()
        cam_fps = PySpin.CFloatPtr(cam.GetNodeMap().GetNode('AcquisitionFrameRate'))
        # Set acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                    Aborting... \n' % i)
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        if consoleEmail.DEBUG:
            print('Camera %d acquisition mode set to continuous...' % i)

        # Begin acquiring images
        cam.BeginAcquisition()

        if consoleEmail.DEBUG:
            print('Camera %d started acquiring images...' % i)

        print()

    # Retrieve, convert, and save images for each camera
    t = {}
    barrier = threading.Barrier(num_cameras + 1)
    video_writers = {}

    for i, cam in enumerate(cam_list):
        # Acquire image on all cameras
        # t[i] = multiprocessing.Process(target=acquire_image, args=(cam_list[i], queues[i], images[i], event_for_main[i], i))

        t[i] = Thread(target=acquire_image,
                      args=(cam_list[i], queues[i], images_queues[i], barrier, i, executor, queue_avi))
        logging.debug("Acquire image --- count:")
        logging.debug(threading.active_count())
        t[i].start()

        width = cam.Width()
        height = cam.Height()

        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()
        # result &= save_img(images[i])
        avi_renderer = get_video_writers(directories['trial'], nodemap, nodemap_tldevice, width, height)
        video_writers[i] = avi_renderer
    # start writing images into a video file
    if queue_avi:
        avi_threads = {}

        for i, cam in enumerate(cam_list):
            avi_threads[i] = Thread(target=create_avi_from_queue, args=(video_writers[i], images_queues[i]))
            avi_threads[i].start()
            logging.debug("start creating the AVI file (video)")

    # thread for Pain Detection
    pain_thread = {}
    for i, cam in enumerate(cam_list):
        if consoleEmail.camera_for_algo == False:  ## Checking for consoleEmail.camera_for_algo: if True, then both cameras needed, If False, only one camera. -- only for pain_thread
            if i == 1:
                break
        # pain_thread[i] = multiprocessing.Process(target=detect_pain, args=(images[i], i))
        pain_thread[i] = Thread(target=detect_pain, args=(i, cam_fps.GetValue(), directories['csv']))
        logging.debug("PAIN --- count:")
        logging.debug(threading.active_count())
        pain_thread[i].start()

    while (True):
        barrier.reset()
        try:
            time_start = time.time()
            # time.sleep(
            #     1 / 30)  # 30 frames per second # int(consoleEmail.frame_rate) for taking 1 picture per frame_rate ie frame_rate = 30 means 1 picture every 1/30 seconds

            for i, cam in enumerate(cam_list):
                # Adding 1 for every frame to be captured
                queues[i].put("1")
                # print("\nAdding 1 to the queue for cam %d" % i)

            # for i, cam in enumerate(cam_list):
            #     # Waiting for each thread to signal successful frame capture
            #     event_for_main[i].wait()
            # logging.debug("cam {}".format(i))
            barrier.wait()
            stop_time = time.time() - time_start
            try:
                logging.debug(f'Synced Frame rate: {1 / stop_time} ')
            except:
                pass
            # logging.debug(f'number of tasks in the executor: {executor}')

        except KeyboardInterrupt:
            consoleEmail.key_press = True
            logging.debug("Stopped by user...")
            print("Stopped by user...")
            break
    logging.debug("Finishing the processes...")
    print("Finishing the processes...")
    # Ending the thread by adding a None in the queue
    for i, cam in enumerate(cam_list):
        queues[i].put(None)
        images_queues[i].put(None)  # end of capturing and writing video file

    # making sure the video generation by frame compilation only starts after all the frames are captured

    for i, cam in enumerate(cam_list):
        if consoleEmail.camera_for_algo == False:  ## Checking for consoleEmail.camera_for_algo: if True, then both cameras needed, If False, only one camera. -- only for pain_thread
            if i == 1:
                break
        logging.debug("waiting for other threads to join (finishing jobs)")
        try:
            t[i].join(timeout=10)
            pain_thread[i].join(timeout=5)
            if queue_avi:
                avi_threads[i].join()
        except:
            pass

    print('complete the capturing and pain detection... \n')
    logging.debug("End of the online section...")
    logging.debug("--------------------------------------")
    logging.debug("Strar creating the video files. This may take time.")
    print("waiting for other threads to stop...")
    if not queue_avi:
        threads_avi = {}
        for i, cam in enumerate(cam_list):
            threads_avi[i] = Thread(target=read_saved_fram_to_avi, args=(i, cam,))
            # read_saved_fram_to_avi(i, cam)
            threads_avi[i].start()

    print("threads done...")
    executor.shutdown(wait=True)
    print("executors done...")
    # Deinitialize each camera
    for i, cam in enumerate(cam_list):
        # reading files from the disk for cam [i]
        # images = read_saved_fram_to_avi(i, cam)

        # width = cam.Width()
        # height = cam.Height()
        # End acquisition
        cam.EndAcquisition()

        # nodemap_tldevice = cam.GetTLDeviceNodeMap()
        # Retrieve GenICam nodemap
        # nodemap = cam.GetNodeMap()
        # result &= save_img(images[i])
        # result &= save_list_to_avi(nodemap, nodemap_tldevice, images, width, height)

        # Deinitialize camera
        cam.DeInit()

        # Release reference to camera
        del cam

    # End of run_multiple_cameras related code block
    #
    #

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    # Send email by calling function send_email
    # send_email()
    # print("Email Sent")

    print("*** End of program ***")
    deinit()  # colorama

    input('Done! Press Enter to exit...')
    return result


numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


from tqdm import tqdm


def read_saved_fram_to_avi(cam_id, cam):
    images = []
    avi_filename = str(consoleEmail.file_name) + '-Uncompressed-Cam%s' % cam_id
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'test_vid_{cam_id}.avi', fourcc, cam.AcquisitionFrameRate(),
                          (cam.Width(), cam.Height()),
                          True)  # The last argument should be True if you are recording in color.
    path = target_images_folder + "/ " + str(cam_id)
    logging.debug("start creating video file for Cam {}.".format(cam_id))
    for filename in tqdm(sorted(glob(path + '\*.jpg'), key=numericalSort)):
        try:
            img = cv2.imread(filename)
            out.write(img)
        except:
            pass
    out.release()
    return images


#####################################################################################################################
if __name__ == '__main__':
    # freeze_support()
    # image = cv2.imread("results/test/captured_images/Cam 0/Image 140.jpg")
    # print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
    main(QUEUE_AVI)
    # detect_pain(0, 30, "results/test/records.csv")
    print("Done!")
