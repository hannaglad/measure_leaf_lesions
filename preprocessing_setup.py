import sys
from os import mkdir
'''
## LEGCOMP
# Directory setup
working_directory = "/data1/hanna/"
# Parent folder of subdirectories containing images
input_prefix = "All_Qst_Leaf_images/"
# Log file to report problems with full images
logfile = "/data1/hanna/ac2_v2.git/preprocessing/unprocessed_files.log"
# ouput directory for split images
outprefix = working_directory+"split_images_2/"
'''


## LOCAL
# Directory setup
working_directory = "/home/hanna/Documents/image_stuff/Original_Image_Data/"
# Parent folder of subdirectories containing images
input_prefix = "All_Qst_Leaf_images/"

# Log file to report problems with full images
logfile = "./unprocessed_files.log"
# ouput directory for split images
outprefix = working_directory+"split_images_for_vae_v2/"

# FILE THAT CONTAINS ISOLATE NAMES FOR MAPPING
name_map="./names_to_qr_codes.csv"
# SET REFERENCE IMAGE TO FIT CONTRAST
reference_image_path = "./reference_image.jpg"


try :
    mkdir(outprefix)
except FileExistsError:
    pass

### FOR TESTING PURPOSES ##
#test_img = str(working_directory+input_prefix+"/1011/1011_r3_a15_4.jpeg")

test_img = "/home/hanna/Documents/image_stuff/Original_Image_Data/All_Qst_Leaf_images/5254/5254_r3_a15_1.jpeg"

# Paramaters to correctly remove black lines from the outlines of the individual boxes (may need to be optimized per dataset)
X_ADD_RIGHT = 20
X_REMOVE_LEFT = 50
Y_ADD = 5

#Self-explanatory
NB_LEAVES_PER_PAGE = 8

# Parameters to calculate and correct for different image resoluations (also should not change unless paper size changes)
# This is the length in cm of the space between the two reference points at the top of the page
KNOWN_X_LENGTH = 170
BOX_CONSTANT_WIDTH = 180
BOX_CONSTANT_HEIGHT = 35

# How big should the final picture be ?
HEIGHT = 300
WIDTH = 3950

# Potentially this is the rescaling issue ...
# Aniks original pictures are 9921x14028px full page
#8068px box width
#1614 box height

X = 9921
Y = 14028

# How many pixels around the qr code to also mask
QR_EXTRA_BORDER = 30

#Self-explanatory
NB_LEAVES_PER_PAGE = 8
