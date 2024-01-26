import sys
import cv2
import numpy as np  
import logging

#To log to file:
#logging.basicConfig(level = logging.DEBUG, filename = 'logs/pgcam.log', filemode = 'w') #set to 'a' to append
#To log to stdout:
logging.basicConfig(level = logging.INFO)  #set to DEBUG if you want way too much info

try:
    import PySpin
except ImportError:
    logging.critical("Pyspin not found. There is no sense going on.")
    raise ImportError


class PgCamera:
    """Class for simple control of a Point Grey camera.

    Uses Spinnaker API. Module documentation `here
    <https://www.flir.com/products/spinnaker-sdk/>`_.
    """
    def __init__(self, roi, frame_rate, gain, exposure):
        self.system = PySpin.System.GetInstance()
        self.cam = self.system.GetCameras()[0]
        self.roi = roi
        self.frame_rate = frame_rate
        self.exposure = exposure
        self.gain = gain
        assert isinstance(self.cam, PySpin.CameraPtr)
        logging.info("Working with a {0} camera".format(self.cam_info()))

    def cam_info(self):
        nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        camera_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceDisplayName'))
        self.camera_name = camera_name.ToString()
        return self.camera_name  
        
    def open_camera(self):
        self.cam.Init()
        nodemap = self.cam.GetNodeMap()
        
        ####################################
        # SET TO CONTINUOUS ACQUISITION MODE
        logging.info("ACQUISITION MODE ")
        acquisition_mode_node = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsAvailable(acquisition_mode_node) and PySpin.IsWritable(acquisition_mode_node):
            logging.debug("    acquisition_mode available")
        else:
            logging.warning("    acquisition_mode is NOT available/writeable.")

        # Retrieve entry node from enumeration node
        acquisition_mode_continuous_node = acquisition_mode_node.GetEntryByName("Continuous")
        if PySpin.IsAvailable(acquisition_mode_continuous_node) and PySpin.IsReadable(acquisition_mode_continuous_node):
            logging.debug("    continuous_mode readable")
        else:
            logging.warning("    continuous_mode NOT available/readable")

        try:
            acquisition_mode_continuous = acquisition_mode_continuous_node.GetValue()
            acquisition_mode_node.SetIntValue(acquisition_mode_continuous)
            logging.info("    Acquisition Mode successfully set to Continuous.\n")
        except Exception as ex:
            logging.error("    acquisition mode not set: {0}.".format(ex))

        ###########
        # ROI: Set this first as frame rate limits depend on this
        if self.roi[0] >= 0:
            try:
                logging.info("ROI")
                #Width 
                #Note set width/height before x/y offset becuase upon initialization max offset is 0 b/c 
                # it is assuming full-frame
                width_node = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
                if PySpin.IsAvailable(width_node):
                    logging.debug("    ROI width_node available.")
                else:
                    logging.debug("    ROI width_node NOT available.")
                    
                if PySpin.IsWritable(width_node):
                    logging.debug("    ROI width_node is writeable")
                else:
                    logging.debug("    ROI width_node NOT writeable.")
                    
                #Regardless, try to set
                #width_to_set = width_node.GetMax()    #default
                width_inc = width_node.GetInc()
                width_to_set = self.roi[2]
                if np.mod(width_to_set, width_inc) != 0:
                    width_to_set = (width_to_set//width_inc)*width_inc
                    logging.warning("    Need to set width in increments of {0}, resetting to {1}.".format(width_inc, width_to_set))
                try:
                    width_node.SetValue(width_to_set)
                    logging.debug('    Width set to {0}.'.format(width_node.GetValue()))           
                except Exception as ex:
                    logging.error(    "ROI width not set: {0}.".format(ex))
                    
                # Height
                height_node = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
                if PySpin.IsAvailable(height_node):
                    logging.debug("    ROI height node available.")
                else:
                    logging.debug("    ROI height node NOT available.")
                                  
                if PySpin.IsWritable(height_node):
                    logging.debug("    ROI height node is writeable.")
                else:
                    logging.debug("    ROI height node is NOT writeable")
                            
                #Regardless, try to set height
                #height_to_set = height_node.GetMax()  #default
                height_inc = height_node.GetInc()
                height_to_set = self.roi[3]
                if np.mod(height_to_set, height_inc) != 0:
                    height_to_set = (height_to_set//height_inc)*height_inc
                    logging.warning("    Need to set height in increments of {0}, resetting to {1}.".format(height_inc, height_to_set))
                try:
                    height_node.SetValue(height_to_set)
                    logging.debug('    Height set to {0}'.format(height_node.GetValue()))
                except Exception as ex:
                    logging.error(    "ROI height not set: {0}.".format(ex))

                
                # x-offset
                offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
                if PySpin.IsAvailable(offset_x_node):
                    logging.debug("    ROI x available")
                else:
                    logging.debug("    ROI x NOT available")
                if  PySpin.IsWritable(offset_x_node):
                    logging.debug("    ROI x writeable")
                else:
                    logging.debug("    ROI x NOT writeable")
                #Try to set
                #x_to_set = offset_x_node.GetMin()  #default (usually 0)
                x_to_set = self.roi[0]
                try:
                    offset_x_node.SetValue(x_to_set)
                    logging.debug('    x offset set to {0}'.format(offset_x_node.GetValue()))
                except Exception as ex:
                    logging.error("    x offset not set: {0}.".format(ex))

                # y-offset
                offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
                if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
                    #y_to_set = offset_y_node.GetMin()  #default (usually 0)
                    #print("    Min, max y: ", offset_y_node.GetMin(), offset_y_node.GetMax())
                    y_to_set = self.roi[1]
                    offset_y_node.SetValue(y_to_set)
                    logging.debug('    y offset set to {0}'.format(offset_y_node.GetValue()))
                else:
                    logging.warning('    ROI: Offset Y not available...')
                    return False
                logging.info('    ROI successfully set to {0} [x, y, width, height]\n'.format(roi))
            except Exception as ex:
                logging.error('    Could not set ROI. Exeption: {0}.'.format(ex))
              
        #############################
        # FRAME RATE : do second (exposure time limits depend on this)

        #Set frame rate value
        logging.info("FRAME RATE")
        self.acquisition_rate_node = self.cam.AcquisitionFrameRate
        rate_max = self.acquisition_rate_node.GetMax()
        rate_min = self.acquisition_rate_node.GetMin()
        logging.info("    Frame rate min|max: {0} | {1}.".format(rate_min, rate_max))
        #Try setting frame rate
        if self.frame_rate > rate_max:
            logging.warning("Attempt to set fps greater than max, setting to {0}".format(rate_max))
            self.frame_rate = rate_max
        elif frame_rate < rate_min:
            logging.warning("Attempt to set fps less than min, setting to {0}".format(rate_min))
            self.frame_rate = rate_min

            
        #First try older, simpler way (will not work on newer cameras for some reason)
        node_acquisition_frame_rate_control_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(node_acquisition_frame_rate_control_enable):
             self.cam.AcquisitionFrameRateEnable.SetValue(True)
        else: #have to do it new way 
            # First disable auto-frame rate
            node_acquisition_frame_rate_control_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))
            if PySpin.IsAvailable(node_acquisition_frame_rate_control_enable) and PySpin.IsWritable(node_acquisition_frame_rate_control_enable):
                logging.debug("    frame rate control available and writeable")
            else:
                logging.warning("    frame rate control not available and writeable")
            try:
                node_acquisition_frame_rate_control_enable.SetValue(True)
            except Exception as ex:
                logging.error("    Could not set acquistion frame rate: {0}".format(ex))
            frame_rate_auto_node = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionFrameRateAuto"))
            if PySpin.IsAvailable(frame_rate_auto_node) and PySpin.IsWritable(frame_rate_auto_node):
                logging.debug("    frame_rate_auto writeable/available")
            else:
                logging.debug("    frame_rate_auto NOT writeable/available")
            try:
                node_frame_rate_auto_off = frame_rate_auto_node.GetEntryByName("Off")
                logging.debug("    got frame_auto_node off")
            except Exception as ex:
                logging.error("   could not get frame_auto_node off: {0}".format(ex))
            if PySpin.IsAvailable(node_frame_rate_auto_off) and PySpin.IsReadable(node_frame_rate_auto_off):
                logging.debug("    frame_rate_auto available/readable")
            else:
                logging.debug("    frame_rate_auto NOT available/readable") 
            try:
                frame_rate_auto_off = node_frame_rate_auto_off.GetValue()
                frame_rate_auto_node.SetIntValue(frame_rate_auto_off)     
                logging.debug( "    Frame Rate Auto set to Off...")
            except Exception as ex:
                logging.error("    Frame_rate_aut could not be turned off: {0}.".format(ex))
                    
            # Second, enable frame rate control
            try:
                enable_rate_mode = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))
                logging.debug("    frame rate mode enabled acquired")
            except Exception as ex:
                logging.error("    could not acquire frame rate enable mode: {0}".format(ex))
                    
            if PySpin.IsAvailable(enable_rate_mode):
                logging.debug("    rate_mode_enable available")
            else:
                logging.debug("    rate_mode_enable NOT available")
                
            if PySpin.IsWritable(enable_rate_mode):
                logging.debug("    rate_mode_enable writeable")
            else:
                logging.debug("    rate_mode_enable NOT writeable")
            try:
                enable_rate_mode.SetValue(True)
                logging.debug("    enable_rate_mode set to True")
            except Exception as ex:
                logging.error("    enable_rate_mode could not be set: {0}.".format(ex))
            # Check to make sure you successfully made frame rate writeable
            if self.acquisition_rate_node.GetAccessMode() == PySpin.RW:  
                logging.debug("    Frame rate mode is read/write.")
            else:
                logging.warning("    frame rate node NOT read/write mode.")
                
        #Frame rate should now be writeable. Set it.
        try:   
            self.acquisition_rate_node.SetValue(self.frame_rate)  
            logging.info("    Frame rate successfully set to {0} Hz\n".format(self.frame_rate))
        except Exception as ex:
            logging.error("    Frame rate not set: {0}.".format(ex))
            
        ##############
        # EXPOSURE 
        logging.info("EXPOSURE")
        #Turn off auto exposure
        exposure_auto_node = self.cam.ExposureAuto
        if exposure_auto_node.GetAccessMode() == PySpin.RW:
            logging.debug("    exposure_auto_node is in read/write mode")
        else:
            logging.debug("    exposure_auto_mode is NOT in read/write mode")
        try:
            exposure_auto_node.SetValue(PySpin.ExposureAuto_Off)
            logging.debug('    Automatic exposure disabled...')
        except Exception as ex:
            logging.error("    Autoexposure Not turned off: {0}".format(ex))
        
        # Check for availability/writeability of exposure time
        exposure_time_node = self.cam.ExposureTime
        if exposure_time_node.GetAccessMode() == PySpin.RW:
            logging.debug('    exposure time is r/w')
        else:
            logging.debug('    exposure time is NOT r/w')
        
        # Set exposure time (and ensure doesn't exceed max)
        exposure_max = exposure_time_node.GetMax()
        exposure_min = exposure_time_node.GetMin()
        logging.info("    min/max exposure times (ms): {0} | {1}".format(exposure_min/1000, exposure_max/1000))

        # camera wants exposure in us:
        exposure_time_to_set = self.exposure*1000  #convert to microseconds
        if exposure_time_to_set > exposure_max:
            logging.warning("    exposure time greater than max: setting it to max of {0}".format(exposure_max/1000))
            exposure_time_to_set = exposure_max
        elif exposure_time_to_set < exposure_min:
            logging.warning("    exposure time less than min: setting it to min of {0}".format(exposure_min/1000))
            exposure_time_to_set = exposure_min
        try:
            exposure_time_node.SetValue(exposure_time_to_set)
            logging.info("    Exposure sucessfully set to {0} ms\n".format(exposure_time_to_set/1000))
        except Exception as ex:
            logging.error("    Exposure NOT set: {0}".format(ex))
        
        #####################################
        # GAIN 
        logging.info("GAIN")
        #Turn off auto-gain
        gain_auto_node = self.cam.GainAuto
        if gain_auto_node.GetAccessMode() == PySpin.RW:
            logging.debug("    autogain in rw mode")
        else:
            logging.warning("    autogain is NOT in rw mode")
        try:
            gain_auto_node.SetValue(PySpin.GainAuto_Off)
            logging.debug("    Automatic gain disabled...")
        except Exception as ex:
            logging.error("   Automatic gain NOT disabled: {0}".format(ex))
        # Set new gain
        gain_node = self.cam.Gain
        gain_min = gain_node.GetMin()
        gain_max = gain_node.GetMax()
        logging.info("    Gain: current, min|max: {0}, {1} | {2} ".format(gain_node.GetValue(), gain_min, gain_max))
        gain_to_set = self.gain
        if gain_to_set >gain_max:
            logging.warning("    gain greater than max - setting it to max of {0}".format(gain_max))
            gain_to_set = gain_max
        elif gain_to_set < gain_min:
            logging.warning("    gain less than min - setting it to min of {0}".format(gain_min))
            gain_to_set = gain_min
        if gain_node.GetAccessMode() == PySpin.RW:
            logging.debug("    Gain is in RW mode")
        else:
            logging.debug("    Gain is NOT in rw mode")
        try:
            gain_node.SetValue(gain_to_set)
            logging.info("    Gain successfully set to {0}.\n".format(gain_to_set))
        except Exception as ex:
            logging.error("    Gain not set: {0}".format(ex))

        #  START ACQUISITION
        self.cam.BeginAcquisition()
        msg = "PgCamera object successfully created"
        logging.info(msg)
        return msg
    
    def read(self):
        #get next image: if it is good, return it
        try:
            image_result = self.cam.GetNextImage()
            if image_result.IsIncomplete():
                return
            else:
                image_converted = np.array(image_result.GetData(), dtype="uint8").reshape((image_result.GetHeight(), 
                                                                                           image_result.GetWidth()) );
                image_result.Release()
                return image_converted

        except PySpin.SpinnakerException as ex:
            logging.error("read error: {0}".format(ex))
            return None

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()
        logging.info("Camera released.")

#%%
if __name__ == "__main__":
    """ 
    Test PySpin api/SpinnakerCamera() using opencv.
    grasshopper maxfov/fps: 2048 x 2048/90
    chameleon maxfov: 1288x968/30
    """
    usage_note = "\nCommand line arguments:\n1: To test simple display [default]\n2: simple_tracker\n"
    roi = [50, 50, 544, 544]  #x-offset, y-offset, width, height
    frame_rate = 25.0
    gain = 1
    exposure_ms = 15.
    if len(sys.argv) == 1:
        print(sys.argv[0], ": ", usage_note)
        test_case = '1'
    else:
        test_case = sys.argv[1]
        
    if test_case == '1':
        """ simple PgCamera use case: just show the image """
        logging.info("\n**Testing PgCam()**".format(frame_rate))
        cv2.namedWindow("PgCam", cv2.WINDOW_NORMAL)
        pgCam = PgCamera(roi, frame_rate, gain, exposure_ms)
        pgCam.open_camera()
        while True:
            image = pgCam.read()
            cv2.imshow("PgCam", image)
            key = cv2.waitKey(1)  
            if key == 27: #escape key
                logging.info("Streaming stopped")
                cv2.destroyAllWindows()
                pgCam.release()
                break
            
    elif test_case == '2':
        """ Testing a simple tracking algorithm to show integration with opencv """
        #%% Preparation and gui/parameter setting
        num_bg_images = 50
        max_thresh = 255
        thresh = 100
        def set_thresh(val):
            global thresh
            thresh = val
        gauss_width = 11
        max_width = 50  #to 101
        def set_gauss_width(val):
            global gauss_width
            gauss_width =  val*2+1 if val > 0 else 0
        
        #%% Create cam instance and get one image
        logging.info("\n**Testing PgCam() Tracking**".format(frame_rate))
        pgCam = PgCamera(roi, frame_rate, gain, exposure_ms)
        pgCam.open_camera()
        #%%
        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Threshold", "Stream", 100, max_thresh, set_thresh)
        cv2.createTrackbar("Smoothing", "Stream", 11, max_width, set_gauss_width)
        image = pgCam.read()
        cv2.imshow("Stream", image)
            
    
        #%%  Get background image mean
        logging.info("    Getting {0} bg images.".format(num_bg_images))
        bg_sum = np.zeros(image.shape)
        for image_ind in range(num_bg_images):
            bg_im = pgCam.read()
            cv2.imshow("Stream", bg_im)
            bg_sum += bg_im
            cv2.waitKey(50) 
        background_image = np.uint8(bg_sum/num_bg_images)
        cv2.imshow("Stream", background_image)
                        
        #%% Start tracking
        #name, window, initial value, max, callback, 
        logging.info("    Tracking starting.")
        while True:
            #Capture raw image
            image = pgCam.read()
            #cv2.imshow("Stream", image)
            
            #Do background subtraction
            bg_subtracted = cv2.absdiff(image, background_image); #   np_image - background_image        
            #cv2.imshow("Stream", bg_subtracted)  #yippeee       
            
            #Gaussian smooth and threshold
            if gauss_width > 0:
                blurred = cv2.GaussianBlur(bg_subtracted, (gauss_width, gauss_width),0)
            else:
                blurred = bg_subtracted.copy()
            ret, thresh_blurred = cv2.threshold(blurred, thresh, max_thresh, cv2.THRESH_BINARY)
            #cv2.imshow("Stream", thresh_blurred)
                   
            # Extract contours
            contours, hierarchy = cv2.findContours(thresh_blurred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #Following two if you want to draw contours
            #contoured_image = cv2.drawContours(image, contours, -1, (0,0,0), 2)  #last arg is thickness
            #cv2.imshow("Stream", contoured_image)
            
            # Get center of mass of first contour 
            try:
                moments = cv2.moments(contours[0])
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])       
                cv2.circle(thresh_blurred, (cx, cy), 5, (0, 0, 0), -1);  #image, center, radius, color, thickness (-1=filled)
            except:
                pass
            cv2.imshow("Stream", thresh_blurred)
            #full_data = cv2.hconcat((blurred, dividing_line, thresh_blurred))
            #cv2.imshow("Stream", full_data)
    
            # next extract fish's orientation and draw arrowed line:
            #try also arrowed line  (cv2.arrowedLine)

            #Keypress/close stuff
            key = cv2.waitKey(1)  
            if key == 27: #escape key
                cv2.destroyAllWindows()
                pgCam.release()
                break
        
        