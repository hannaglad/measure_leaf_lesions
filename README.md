This repo contains two scripts I have written to help with image analysis of wheat leaves infected by Zymo. 
It is based on the old ImageJ macro, and takes the same format as input (8 leaves scanned on a generated PDF) 

### Important note
This is far from a "finished" program, but it seems lesion measurements already work a bit better than the old macro. 
I have mainly made this available in case anyone wants to improve the macro later on. 

# How to use
There are two scripts that need to be run consecutively. Preprocessing and then lesion measurement.

## Example 
`python preprocessing.py input_images/ preprocessed_images/ 2`

`python measure_lesions.py preprocessed_images/ results/` 

### 1. preprocessing.py

This script performs the preprocessing; it separates all the individual leaves and removes the background noise.

This script is in a "barely usable" format and will probably require that you optimize a little bit for your dataset. If it doesn't work out of the box, 
all of the important parameters are explained at the top of the script. Most of the time you just need to adjust parameters for the reference point dection. 

Usage : 
`preprocessing.py input_directory output_directory label_fomat`

label_format ; 
* 1: Images are saved to folders by host/replicate
* 2: Images are saved to individual folers. Usefull for debugging, especially with labels like `extra-leaf-2`

### 2. measure_lesions.py 

This script measures the lesions that are present on the image. Run the script with --h to get a detailed explanation of the parameters. 
This script works quite well and won't require much tuning, except the thresholding which can be changed from the command line. 

