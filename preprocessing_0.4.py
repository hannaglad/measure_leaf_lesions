import cv2
import numpy as np
import random 
import string
import imutils
import math
from PIL import Image as im
from skimage import exposure, img_as_float, img_as_ubyte
import pyzbar.pyzbar as pyzbar
import sys
import os
from os import listdir, path, mkdir
from pathlib import Path
import re
from os.path import exists
from datetime import datetime
import matplotlib.pyplot as plt
import time 
import glob


DEBUG=False

#logfile = "./logs/log.txt"
today = datetime.now() 
#failed_img_dir = "./failed_images"

## USAGE 
# python preprocessing_0.4.py INPUT_DIR OUTPUT_DIR 

# CLEAN THIS SHIT 


# TO DO :
# multithread this shit so that is useable.... so... so... so ... slow... 

# ADD MODE FOR MACRO CLEAN UP ::
    # - rewrite extracted leaves and QR codes into original template as to be used as input to existing 
    # IMAGEJ macro 
    # Expected benefits; background removal and contrast normalization!! 

# Maybe add a way to compare total pass rate with an expected pass rate (for example; usr input)
min_leaf_area_threshold = 100000.0
max_leaf_area_threshold = 10000000.0
# Number of individual images per page 
NB_LEAVES_PER_PAGE = 8 

# Expacted size in pixels 
# Aniks original pictures are 9921x14028px full page
X = 12267
Y = 17261

#X = 4960
#Y = 7014

# Parameters for reference circle detection 
min_radius = int(0.006*X)
max_radius = int(0.01*X)
min_radius_2 = int(0.002*X)
max_radius_2 = int(0.005*X)

# Paramaters to correctly remove black lines from the outlines of the individual boxes (may need to be optimized per dataset)
X_ADD_RIGHT = 20
X_REMOVE_LEFT = 50
Y_ADD = 1
KNOWN_X_LENGTH = 170
BOX_CONSTANT_WIDTH = 180
BOX_CONSTANT_HEIGHT = 35
# How many pixels around the qr code to als name = "_".join(decoded[:-1])o mask
QR_EXTRA_BORDER = 5

# FILE THAT CONTAINS ISOLATE NAMES FOR MAPPING
name_map="./names_to_qr_codes.csv"
# SET REFERENCE IMAGE TO FIT CONTRAST
reference_image_path = "./reference_image.jpg"

## Utility functions ##
def log(string, logfile):
    with open(logfile, 'a') as file:
        file.write(string)
        file.write("\n")

def show(image, name="image"):
    cv2.startWindowThread()
    cv2.imshow(name, image)
    key = None
    while key != ord('a'):
        key = cv2.waitKey(0)
    cv2.destroyAllWindows()


def error_save(original_image, img_name, error_code):
    fail_map = {1 :"/ref_points_1/", 2: "/ref_points_2/", 3: "/QR/", 4:"/leaf/"}
    subdir = fail_map[error_code]

    if not os.path.exists(failed_img_dir+subdir+img_name):
        cv2.imwrite(failed_img_dir+subdir+img_name, original_image)

def display_all_contours(contours, image):
    copy = image.copy() 
    for i in range(len(contours)):
        cv2.drawContours(copy, contours[i], -1, (255, 0, 0), 3)
    show(copy)
    return 0 


## Processing functions ##
def detect_ref_points(scaled, show_z=True, radius_params=None):

    if radius_params:
        if DEBUG:
            print(radius_params)
        min_radius, max_radius, min_radius_2, max_radius_2 = radius_params

    failed = False
    ## Detect reference points at the top of the page 
    #top = 300
    top = 500

    top_of_image = scaled[:top, :, :] # Focus on top of the image
    top_grayscale = cv2.cvtColor(top_of_image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    top_grayscale = cv2.blur(top_grayscale, (3,3)) # Blur to remove noise 

    circles = cv2.HoughCircles(top_grayscale, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius) # Detect circles
    if DEBUG:
        print(circles)
    # If less than 2 circles detected, save imag e to failed circles folder 
    if circles is None or circles.shape[1] < 2 :
        failed = True 
        
    else : 
        circles = np.round(circles[0, :].astype("int"))
        xlist = []
        ylist = []
        if show_z:
            copy = top_of_image.copy() 
        
        for x, y, r in circles:
            # Use the radius of the circle to make sure we have detected the correct points
            # The radius of the referce circles at a downscale facor of 2 is 59
            if r>= min_radius_2 and r<=max_radius_2:
                xlist.append(x)
                ylist.append(y)
                if show_z:
                    cv2.circle(copy, (x,y),r, (0, 255, 0), 4)
        if show_z:
            show(copy)
        
        # If more than two circles detected, save to failed circles folder 
        if len(xlist) < 2 or len(ylist) < 2:
            print([(x,y,r) for x,y,r in circles])
            failed = True 
        
         # Sort by x-coordinate so we don't acidently invert the angle for straightening
        elif xlist[0] > xlist[1]:
            tmpx = xlist[0]
            tmpy = ylist[0]
            xlist[0] = xlist[1]
            ylist[0] = ylist[1]

            xlist[1] = tmpx
            ylist[1] = tmpy 
    
    if failed == True : 
        return None
    else : 
        return ([xlist[0], ylist[0]], [xlist[1], ylist[1]])


def scale_intensity(target_grayscale):
    '''
    Function to scale the color intensity of an image based on a reference image with perfect contrast
    Input : Grayscale image
    Output : Corrected grayscale image
    '''
    ## Using a reference image with perfect contrast
    reference_grayscale = cv2.imread(reference_image_path, 0)
    reference_grayscale = img_as_float(reference_grayscale)
    target_grayscale = img_as_float(target_grayscale)
    max_val = np.max(reference_grayscale)
    min_val = np.min(reference_grayscale)
    target_rescaled = exposure.rescale_intensity(
        target_grayscale, out_range=(min_val, max_val))
    target_rescaled = img_as_ubyte(target_rescaled)
    return target_rescaled

def detect_QR(grayscale, name_map=None, show_zone=False):
    '''
    Detects QR code in a box
    Input :
        grayscale : grayscale box
        name_nap = file containing alternative names for each isolate; to ensure consistent naming
    '''

    # REDUCE THE SEARCH SPACE ; VASTLY IMPROVES RESULT
    detection_zone = grayscale[:, 0:400]

    if show_zone:
        show(detection_zone)

    objects = pyzbar.decode(detection_zone)
   

    if len(objects) > 0 and len(objects) < 2:
        rect = objects[0].polygon
        decoded = str(objects[0].data)

        decoded = decoded.strip(".jpg")
        decoded  = decoded.strip("b'")
        decoded = decoded.split("_")

        host = decoded[-1]
        test = decoded[2]

        if "leaf" in test:
            replicate = "_".join(decoded[2:4])
            search_string = "_".join(decoded[0:2])
        else:
            replicate = "_".join(decoded[3:5])
            search_string = "_".join(decoded[0:3])

        if name_map :
            with open(name_map, 'r') as map: # Search for corresponding name in name_map file
                for line in map.readlines():
                    l = line.strip().split(',') 
                    if search_string in l[1]: # If found in name map, take the correct name 
                        name=str(l[0])
                        break
                    else:
                        name = "_".join(decoded[:-1])
        else : 
            name = "_".join(decoded[:-1])
                   
        return [host, name, replicate, rect]

    else:
        return None



def log_non_fatal(failed_indexes, image_name, logfile, error_code):
    code_map = {3 : "QR dection", 4: "Leaf extraction"}
    for idx in failed_indexes:
        logstring = img_name+" pos " + str(idx) + str(code_map[error_code])
        log(logstring, logfile)
    


def process_image(image_path, logfile, name_map=None, show_ref_points=False, show_QR=True, show_contours=True):
    '''
    Applies preprocessing pipeline to A4 scan of wheat leaves 
    Input :  RAW A4 scan of 8 leaves
    Output : Separated and extracted leaf contours with name, host and replicate information
    '''

    # Define list for output
    output = []

    # Get name of image from path 
    img_name = image_path.split("/")[-1]

    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR) # Read input image 

    # Get size and radius for reference points
    X = np.shape(original_image)[1]
    Y = np.shape(original_image)[0]
    min_radius = int(0.006*X)-1
    max_radius = int(0.01*X)+1
    min_radius_2 = int(0.002*X)
    max_radius_2 = int(0.007*X)
    radius_params = [min_radius, max_radius, min_radius_2, max_radius_2]

    scaled = cv2.resize(original_image, (int(X/2), int(Y/2)), interpolation=cv2.INTER_AREA) # Downscale by 1/2
    ref_points = detect_ref_points(scaled, show_z=show_ref_points, radius_params=radius_params) # Get ref points from top 
    
    # Check if ref point detection has worked 
    if not isinstance(ref_points, tuple):
        cv2.imwrite(failed_img_dir+"/ref_points_1/"+img_name, scaled)
        error_save(original_image, img_name, 1)
        logstring = img_name+" : Ref point detection 1 "
        log(logstring, logfile)
    
    else : 

        #------------ Straighten image  ----------------------------------------
        # Calculate angle
        x1 = ref_points[0][0]
        x2 = ref_points[1][0]
        y1 = ref_points[0][1]
        y2 = ref_points[1][1]

        slope = (y2-y1)/(x2-x1)
        angle = math.degrees(math.atan(slope))

        # rotate
        rotated = imutils.rotate(scaled, 0+angle)

        #-------------- Crop image ------------------------------------------------
        # Re-detect reference points for cropping
        new_ref_points = detect_ref_points(rotated, show_z=False, radius_params=radius_params)
        if not isinstance(new_ref_points, tuple):
            cv2.imwrite(failed_img_dir+"/ref_points_2/"+img_name, scaled)
            error_save(original_image, img_name, 2)
            logstring = img_name+" : Ref point detection 2 "
            log(logstring)
        
        else:
            x1 = new_ref_points[0][0]
            x2 = new_ref_points[1][0]
            y1 = new_ref_points[0][1]
            y2 = new_ref_points[1][1]

            # Crop
            cropped = rotated[:, x1+X_ADD_RIGHT:x2-X_REMOVE_LEFT, :]
       
            # Calculate the resolution of the image
            observed_x_length = np.sqrt((x2-x1)**2)+((y2-y1)**2)
            resolution = observed_x_length/KNOWN_X_LENGTH
            box_height = int(BOX_CONSTANT_HEIGHT*resolution)
            
            # Start from the first horizontal line and not the middle of the reference point
            ypos = int(y1+5*resolution)

            # -------------- PER-BOX OPERATIONS  ------------------------------------------------
            for i in range(NB_LEAVES_PER_PAGE):
                
                # Calculate where we need to end the vertical crop
                yend = int(ypos+box_height)
                # Crop the box
                current_box = cropped[ypos:yend, :, :]
                
                if i == 0:
                 show(current_box)
                
                # Add to current ypos to avoid taking the black outline in the next box
                ypos = yend+Y_ADD

                #-------------- QR READING  ------------------------------------------------
                # CONVERT TO AN INITIAL GRAYSCALE
                try :
                    current_box_grayscale = cv2.cvtColor(current_box, cv2.COLOR_BGR2GRAY)
                except : 
                    raise SystemExit("Issue in grayscale conversion on {}. Exiting.".format(img_name))
                
                QR_output = detect_QR(current_box_grayscale, name_map, show_QR)
                
                if not isinstance(QR_output, list):
                    cv2.imwrite(failed_img_dir+"/QR/+img_name+index_"+str(i)+".png", current_box)
                    error_save(original_image, img_name, 3)
                    logstring = img_name+" pos " + str(i) +"  : QR detection"
                    log(logstring, logfile)
                
                else : 
                    host, name, replicate, rect = QR_output

                    # Get the rectangle of the QR code for masking
                    x_start = min(0, (rect[0].x - QR_EXTRA_BORDER))
                    x_end = rect[2].x + QR_EXTRA_BORDER
                    y_start = min(0,(rect[0].y - QR_EXTRA_BORDER))
                    y_end = rect[2].y + QR_EXTRA_BORDER

                    # Create a mask 
                    mask = np.zeros_like(current_box_grayscale)
                    cv2.rectangle(mask, (x_start,y_start), (x_end, y_end), 1, -1)

                    # Set pixels of QR code to median value of image    
                    current_box_grayscale[mask==1] = np.median(current_box_grayscale)

                    #-------------- LEAF EXTRACTION  ------------------------------------------------
                    # Use reference image to adjust contrast 
                    current_adjusted = scale_intensity(current_box_grayscale)

                    # FIND MIN VALUE FOR THRESHOLD
                    # Median because most pixels are white.
                    # Don't ask why -20. I dont know. But it works. I guess because the median value is slightly pushed to darker colors by the leaf and qr codes
                    t_min = np.median(current_adjusted) - 20        
                    # Binarize images
                    r, binary = cv2.threshold(current_adjusted, t_min, 250, cv2.THRESH_BINARY_INV)

                    # find countours
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                    # Show all detected contours 
                    #display_all_contours(contours, current_box)

                    # Get the largest contour which should be the leaf
                    contours = sorted(contours, key=cv2.contourArea)   

                    largest = contours[len(contours)-1:]
                    
                    if show_contours:
                        box_copy = current_box.copy()
                        cv2.drawContours(box_copy, largest, -1, (255, 0, 0), 3)
                    
                    # Check that the area of the contours isn't too big. If it is, take the next one
                    while cv2.contourArea(largest[0]) >= max_leaf_area_threshold:
                        contours.pop(len(contours)-1)
                        largest = contours[len(contours)-1:] 
                        
                        if show_contours:
                            box_copy = current_box.copy()
                            cv2.drawContours(box_copy, largest, -1, (255, 0, 0), 3)
                            show(box_copy)
                    
                    # If we end up with a contour that is smaller than said threshold, then we haven't
                    # extracted the leaf... so fail... 
                    if cv2.contourArea(largest[0]) < min_leaf_area_threshold:

                        if not show_contours:
                            box_copy = current_box.copy()
                            cv2.drawContours(box_copy, largest, -1, (255, 0, 0), 3)

                        cv2.imwrite(failed_img_dir+"/leaf/"+img_name+"index_"+str(i)+".png", box_copy)
                        error_save(original_image, img_name, 4)
                        logstring = img_name+" pos " + str(i) +"  : Leaf extraction"
                        log(logstring, logfile)
                    
                    else:
                        x,y,w,h = cv2.boundingRect(largest[0])
                        mask = np.zeros_like(current_box, dtype=np.uint8)
                        cv2.fillPoly(mask, largest, (255, 255, 255))
                        final = current_box.copy()
                        final[mask == 0, ] = 255
                        final = final[y:y+h, x:x+w]
                        box_output = [final, host, name, replicate]
                        output.append(box_output)

    nb_of_outputs = len(output)
    print("{}/8 leaves sucessfully extracted from {}".format(nb_of_outputs, img_name))
    return (output)


def unit_test(image_path):
    process_image(image_path, show_ref_points=True, show_QR=True, show_contours=True, logfile="./testlog")


'''
failed_img_dir = str("/home/hanna/Documents/Luzia_images/"+"failed_images")
path = "/home/hanna/Documents/Luzia_images/Leave_scans/1_3D7_PF.jpg"
unit_test(path)
'''


#test_path = "/home/hanna/Documents/image_stuff/Original_Image_Data/All_Qst_Leaf_images/1204/1204_r1_a12_1.jpeg"
#unit_test(test_path)




if __name__ == "__main__":


    ## SET UP 
    # Get input directory 
    try:
        input_dir = sys.argv[1]
    except IndexError:
        raise SystemExit("No input directory specified")
    # Create input directory if does not exist
    if not os.path.isdir(input_dir) or not exists(input_dir):
        raise SystemExit("Please check input path")

    # Get output directory
    try:
        output_dir = sys.argv[2]
    except IndexError:
        raise SystemExit("No output directory specified")

    # Correct outpath if missing slash
    if not output_dir.endswith("/"):
        output_dir = output_dir+"/"

    # Create outpath if does not exist
    if not os.path.isdir(output_dir):
        print("Output directory does not exist. Creating folder ...")
        os.mkdir(output_dir)


    full_seen = 0 
    failed_full = 0 
    passed_boxes = 0
    start_global = time.perf_counter() 
    image_times = []

    logfile = str(output_dir+"preprocessing_log.txt") 
    failed_img_dir = str(output_dir+"failed_images")

    print("******************************************")
    print("IMAGE PRE-PROCESSING PIPELINE FOR VAE-GWAS")
    print("Hanna Glad, July 2022") 
    print("Laboratory of Evolutionary Genetics, UNINE")
    print("******************************************")


    if not os.path.exists(logfile):
        with open(logfile, 'x') as file:
            pass 
    else : 
        print("\n{} exists".format(logfile))
        usr_inp = input("Would you like to clear it ? y/[N] ")
        if usr_inp == "y":
            file = open(logfile, "w")
            file.close()
            print("logfile has been cleared.\n") 

    
    if not os.path.isdir(failed_img_dir):
        os.mkdir(failed_img_dir) 
    
    if not os.path.isdir(failed_img_dir+"/ref_points_1"):
        os.mkdir(failed_img_dir+"/ref_points_1")
        os.mkdir(failed_img_dir+"/ref_points_2")
        os.mkdir(failed_img_dir+"/QR")
        os.mkdir(failed_img_dir+"/leaf")
        os.mkdir(failed_img_dir+"/all_full_images")
    
    dirlist = glob.glob(failed_img_dir+"/*/*") 
    if len(dirlist) > 5:
        print("Failed image directory is currently populated")
        usr_inp = input("Would you like to clear it's contents ? y/[N] ") 
        if usr_inp == "y":
            [os.remove(f) for f in dirlist if os.path.isfile(f)] 
            print("Failed image directory has been cleared")
    
    print("-------------------------------------------")
    print("Input folder is : {}".format(input_dir))
    print("Parent output folder is : {}".format(output_dir))
    print("Failed images are available in : {}".format(failed_img_dir))
    print("-------------------------------------------")
     
    log("**** New Run on {} ****".format(today), logfile)
    log(" -- Input Dir : {}".format(input_dir), logfile)
    log(" -- Failed image Dir : {}".format(failed_img_dir), logfile)

    for file in listdir(input_dir):
        if file.endswith(".jpeg") or file.endswith(".jpg"): 
            img_start = time.perf_counter() 
            full_seen += 1 
            filepath = path.join(input_dir, file)
            boxlist = process_image(filepath, logfile, show_ref_points=False, show_QR=False, show_contours=False)
        
            if len(boxlist) == 0:
                print("{} failed.".format(file))
                failed_full += 1 
            else:
                for box in boxlist:
                    img = box[0]
                    host = box[1]
                    name = box[2]
                    replicate = box[3]
                    final_output_dir = str(output_dir + name + "/")
                
                    try:
                        mkdir(final_output_dir)
                    except FileExistsError :
                        pass

                    outname = str(final_output_dir+name+replicate+".jpg")
                    cv2.imwrite(outname, img)
                    passed_boxes +=1 
            img_end = time.perf_counter()
            image_times.append(img_end-img_start)

    print("Done")
    total_expected_boxes = full_seen*8 
    failed_boxes = total_expected_boxes-passed_boxes
    total_failed = failed_boxes + (failed_full*8)
    end_global = time.perf_counter()
    total_time = end_global - start_global
    end = datetime.now() 

    percent_total_passed = 1-(failed_full/full_seen)
    percent_boxes_passed = 1-(failed_boxes/total_expected_boxes) 
    percent_all_passed = 1-(total_failed/total_expected_boxes)

    log("**** Run finished on {}. Total time: {:.2f} s, Per image time: {:.2f} s **** \n".format(end, total_time, np.mean(image_times)), logfile)

    print("******************************************")
    print("Total execution time was {:.2f} seconds".format(total_time))
    print("Mean execution time per image was {:.3f} minutes ({:.2f} seconds).".format(np.mean(image_times)/60, np.mean(image_times)))
    print("Extracted images are available in {}".format(output_dir))
    print("{}/{} full images failed box extraction (pass rate = {:.3f}%)".format(failed_full, full_seen, percent_total_passed*100)) 
    print("{}/{} boxes failed leaf extraction (pass rate = {:.3f}%)".format(failed_boxes, total_expected_boxes, percent_boxes_passed*100))
    print("Overall, {}/{} leaves are missing (global pass rate = {:.3f}%)".format(total_failed, total_expected_boxes, percent_all_passed*100))    
    print("******************************************")

