# Assignment 2 : Image Enhancement and Filtering
# Authors      : Paulo Rocha nusp
#                Victor R. Russo 11218855
# Course       : scc025 - Digital Image Processing
# Period       : 2020 First Semester
#
import numpy as np
import imageio as im


#this functions are some utilitaries used in the calculations
def euclidian_distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**(0.5)

def padding_image(input_img, filter_size):
    adjustment=int(np.floor(filter_size/2))
    return np.pad(input_img,pad_width=adjustment,mode='constant')

def gaussian_kernel(sigma, x):
    return (1/(2*np.pi*(sigma**2)))*(np.e**(-1*((x**2)/(2*(sigma**2)))))

def spatial_gaussian_component(sigma,size):
    filter = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            filter[i][j] = gaussian_kernel(sigma,euclidian_distance(i-1,0,j-1,0))
    return filter

def convolution(img, f):
    #output image
    out_img = np.zeros(img.shape, dtype=np.uint8)

    #fliping the filter in
    f_flip = np.flip(np.flip(f,0),1)

    #getting the dimensios of the filter
    f_n,f_m= f.shape

    fa = int((f_n-1)/2)
    fb = int((f_m-1)/2)

    #padding the image and getting its dimensions
    pd_img = padding_image(img, f_n)
    pd_img_n,pd_img_m = pd_img.shape

    for x in range(fa, (pd_img_n-fa)):
        for y in range(fb, (pd_img_m - fb)):

            img_region = pd_img[ x-fa:x+(fa+1), y-fb:y+(fb+1)]

            out_img[(x-fa),(y-fb)] = np.sum(np.multiply(img_region, f_flip))

    return out_img.astype(np.uint8)

def normalization(img):
    m = (img.astype(np.int32)).min()
    M = (img.astype(np.int32)).max()
    return (((img-m)*255)/M).astype(np.uint8)

#this functions gets the parameters for each method and calls for the tranformation itself
def apply_bilateral_filter(input_img):
    filter_size = int(input())
    padded_image = padding_image(input_img,filter_size)
    sigma_s = float(input())
    sigma_r = float(input())
    filter = spatial_gaussian_component(sigma_s,filter_size)
    #return bilateral_filter(input_img,sigma_row,sigma_col)
    return padded_image

def apply_laplacian_filter(input_img):
    kernel1 = np.matrix([[0,-1,0], [-1,4,-1], [0,-1,0]])
    kernel2 = np.matrix([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    c = float(input())
    kn_option = int(input())

    if(kn_option == 1):
        img_aux = convolution(input_img, kernel1)
    else:
        img_aux = convolution(input_img, kernel1)

    img_aux = normalization(img_aux)


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

#apply the correct filter to the image
transformed_image = (methods[method-1])(input_img)

if save_option == 1:
    im.imwrite('output_img.png',transformed_image)
