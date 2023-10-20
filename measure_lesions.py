import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import sys
import os
import matplotlib.pyplot as plt
import progressbar 
import argparse 


# These parameters can be tuned to increase/decrease blurring before we apply the thresholding to detect the lesions. 
# They aren't included as parameters to the script because they are unlikely to change alot, but if you want to play around with them, feel free.
# The blurring mainly reduces the artifacts of illumination caused by the scanner beam.
FIRST_BLUR_KERNEL = (5,5)
SECOND_BLUR_KERNEL = (9,9)
BLUR_ITERATIONS = 10 
GAUSSIAN_BLUR_KERNEL = (11,11)

## The threshold parameter (changeable with --thresh) is the number of bins above the mediam green peak that we want to include as being "normal" tissue. 
# E.g setting to 0 means that any slight deviation from the main "green" color in the image is sick, wherease setting to 10 means that we are including a large range of green colors as being "normal" tissue.


def log_result(lesion_percentage, image_path, logfile="./lesion_measurements.csv"):

    image_name = image_path.split("/")[-1]
    image_name = image_name.replace(".jpg", "")
    text = str(image_name + "," + str(lesion_percentage))
    
    with open(logfile, 'a') as log:
        log.write(text)
        log.write("\n")


def detect_lesions(image_path, out_path=0, save=True, upper_threshold=4):

    # Read in image 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale 
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize grayscaled BGR image 
    _, binary = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Detect contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Get largest contour 
    contours = sorted(contours, key=cv2.contourArea)   
    largest = contours[len(contours)-1:]
    leaf_mask = np.zeros_like(binary)
    cv2.drawContours(leaf_mask, largest, -1, (255),-1)
    # Create boolean mask 
    boolArr = (leaf_mask != 255)

    # Calculate appropriate width of the contour
    leaf_area = cv2.contourArea(largest[0])
    contour_width = int(np.sqrt(leaf_area/np.pi))

    # Create mask of JUST THE contour of the leaf (for blurring later) 
    leaf_outline_mask = np.zeros_like(binary)
    cv2.drawContours(leaf_outline_mask, largest, -1, (255), contour_width)

    # Convert BGR to LAB 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Extract A channel (red-green channel)
    a_channel = lab[:,:,1]

    # Normalize without using the mask (better because takes white which is already the same for each image)
    unmasked_normalized = cv2.normalize(a_channel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Make the background white again 
    unmasked_normalized[boolArr] = 0

    # First median blur to remove salt and pepper noise 
    a_channel_blurred = cv2.medianBlur(unmasked_normalized,11)
    a_channel_blurred = cv2.GaussianBlur(unmasked_normalized, (7,7), cv2.BORDER_DEFAULT)

    # Threshold based on histogram of normalized a channel 
    counts, bins = np.histogram(a_channel_blurred[boolArr==False], bins=40)
    counts_fixed = np.append(counts, 0)

    # Create mask for values that just aren't green 
    mask = ((bins > 0) & (bins < 105))
    max_count_index = np.argmax(counts_fixed[mask])
    # Retreive green peak in original array
    green_peak_idx = np.arange(counts_fixed.shape[0])[mask][max_count_index]
    thresh_down = bins[green_peak_idx]

    # Define upper threshold in terms of bins above the green peak 
    n_bins_above = upper_threshold
    thresh_up = bins[green_peak_idx + n_bins_above]

    # Use the mask to apply only to the desired areas (erosion for contour, dilation for inside)
    leaf_indices = np.where(leaf_mask==255)
    contour_indices = np.where(leaf_outline_mask == 255)

    ## To reduce the noise comming from the apparent illumination at the top of the screen
    # Perform erosion and apply to the leaf contours

    blurred_for_leaf = cv2.erode(a_channel_blurred, kernel=FIRST_BLUR_KERNEL, iterations=BLUR_ITERATIONS)
    
    # To expand real leasions in leaf apply dilation to the inside of the leaf 
    dilated_in_leaf = cv2.dilate(a_channel_blurred, kernel=SECOND_BLUR_KERNEL, iterations=BLUR_ITERATIONS)

    # Apply to the correction locations 
    a_channel_blurred[leaf_indices] = dilated_in_leaf[leaf_indices]
    a_channel_blurred[contour_indices] = blurred_for_leaf[contour_indices]

    # Gaussian blur for noise reduction again 
    a_channel_blurred = cv2.GaussianBlur(a_channel_blurred, GAUSSIAN_BLUR_KERNEL, cv2.BORDER_DEFAULT)

    # THRESHOLD 
    th1 = cv2.threshold(a_channel_blurred, thresh_up, 255, cv2.THRESH_BINARY_INV)[1]
    th2 = cv2.threshold(a_channel_blurred, thresh_down, 255, cv2.THRESH_BINARY_INV)[1]

    res = cv2.add(th1, th2)
    res_inv = 255-res 

    # Find countours to draw on overlay 
    lesion_cnts, _ = cv2.findContours(res_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    overlay = image.copy()  
    cv2.drawContours(overlay, lesion_cnts, -1, (0,255,0), 10)
    

    fig,ax = plt.subplots(2,1) 
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # For debugging the thresholding ... 
    #ax[2].imshow(a_channel_blurred, cmap="gray")
    #ax[3].stairs(counts, bins)
    #ax[3].vlines(x=[thresh_down, thresh_up], ymin=0, ymax=max(counts), linestyles="dashed") 
    #ax[4].stairs(counts_g, bins_g)
    #ax[5].imshow(a_channel, cmap="gray")
    
    if save:
        plt.savefig(out_path)    
        plt.close()
    else : 
        plt.show()
        plt.close()
    

    # Calculate percentage of pixels that are lesions vs healthy 
    # Number of pixels that are white in res-inv are the number of pixels belonging to the lesions 
    lesion_pixels = np.sum(res_inv == 255)
   
    # Number of white pixles in the leaf mask is number of pixels beloning to the leaf 
    leaf_pixels = np.sum(leaf_mask == 255) 
    percent = (lesion_pixels * 100) / leaf_pixels
    percent = round(percent, 6)
    out_string = "{:.2f}/{:.2f}={:.3f}%".format(lesion_pixels, leaf_pixels, percent)

    return percent


def list_files(input_dir):
    filepaths = [] 
    file_basenames = [] 
    subdirs = [x[0] for x in os.walk(input_dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                file_basenames.append(file)
                filepaths.append(os.path.join(subdir, file))
    return filepaths, file_basenames

# Unit test single image 
def ut(test_path, thresh=4):
    detect_lesions(test_path, save=False, upper_threshold=thresh)



if __name__ == "__main__" : 

    
    parser = argparse.ArgumentParser(description = "Detects lesions in images of infected leaves and calculates the percentage of pixels belonging to said lesions")
    parser.add_argument("in_dir", help ="Parent directory of images - should be output from Image pre-processing pipeline")
    parser.add_argument("--outpath", help = "Path to save the lesion overlays")
    parser.add_argument("--thresh", help="Set value of threshold for lesion detection. Default is 4. Increase if non-infected area is included in lesions. Decrease if infected area is missing from lesions. Range: 0-38")
    parser.add_argument("--log", help = "Path to save logfile ; Defaults to outpath/lesion_measurements.csv")
    parser.add_argument("--ut", help = "Run test on single image for debugging", action="store_true")


    args = parser.parse_args()
    

    if not args.thresh:
        upper_threshold = 4
    else : 
        upper_threshold = int(args.thresh)

    # If specified, run only a unit test 
    if args.ut:
        args.outpath = None
        print("Running test on single image. Press q to exit")
        if not os.path.isfile(args.in_dir):
            print("Please check input path")
            sys.exit()
        else:
            image = args.in_dir
            ut(image, upper_threshold)
            sys.exit()


    # Otherwise run on a directory 
    if not os.path.isdir(args.in_dir):
        print("Please check input path")
        sys.exit() 
    
    if not os.path.isdir(args.outpath):
        os.mkdir(args.outpath)

    if not args.outpath.endswith("/"):
        args.outpath = args.outpath+"/"

    if not args.log:
        log = args.outpath+"lesion_measurements.csv"
    else : 
        log = args.log  



    files, names = list_files(args.in_dir)

    # Create progress bar 
    widgets = [progressbar.Bar('*'), progressbar.SimpleProgress()]
    bar = progressbar.ProgressBar(maxval=len(files), widgets=widgets).start() 

    for i in range(len(files)):
        im_outpath = args.outpath + names[i]
        percent = detect_lesions(files[i], out_path=im_outpath, save=True, upper_threshold=upper_threshold)
        log_result(percent, files[i], logfile=log)
        bar.update(i)
    bar.finish()


