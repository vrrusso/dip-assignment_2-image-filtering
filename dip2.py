# Assignment 2 : Image Enhancement and Filtering
# Authors      : Paulo Rocha nusp
#                Victor R. Russo 11218855
# Course       : scc025 - Digital Image Processing
# Period       : 2020 First Semester
#
import numpy as np
import imageio as im


def padding_image(input_img, filter_size):
    adjustment=int(np.floor(filter_size/2))
    return np.pad(input_img,pad_width=adjustment,mode='constant')

#this functions gets the parameters for each method and calls for the tranformation itself
def apply_bilateral_filter(input_img):
    filter_size = int(input())
    padded_image = padding_image(input_img,filter_size)
    sigma_row = float(input())
    sigma_col = float(input())
    #return bilateral_filter(input_img,sigma_row,sigma_col)
    return padded_image

def apply_laplacian_filter(input_img):
    print('laplacian')
    return 0

def apply_vignette_filter(input_img):
    print('vignette')
    return 0

#puts the function in an array, so i can call them only once without if-else structures
methods = [apply_bilateral_filter, apply_laplacian_filter, apply_vignette_filter]



#reads the image
filename = str(input()).rstrip()
input_img = im.imread(filename)

method = int(input())
save_option = int(input())

transformed_image = (methods[method-1])(input_img)

if save_option == 1:
    im.imwrite('output_img.png',transformed_image)


