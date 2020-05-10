# Assignment 2 : Image Enhancement and Filtering
# Authors      : Paulo Rocha nusp
#                Victor R. Russo 11218855
# Course       : scc025 - Digital Image Processing
# Period       : 2020 First Semester
#
#
# gitsource    : https://github.com/vrrusso/dip-assignment_2-image-filtering
#
import numpy as np
import imageio as im
import math


#this functions are some utilitaries used in the calculations
def rse(in_img, out_img):
    k,l = in_img.shape
    s = 0.0
    for i in range(k):
        for j in range(l):
            s += (float(out_img[i][j])-float(in_img[i][j]))**2
    return math.sqrt(s)

def euclidian_distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**(0.5)

def padding_image(input_img, filter_size):
    adjustment=int(np.floor(filter_size/2))
    return np.pad(input_img,pad_width=adjustment,mode='constant')

def gaussian_kernel(sigma, x):
    return (1/(2*np.pi*(sigma**2)))*(np.e**(-1*((x**2)/(2*(sigma**2)))))

def vignette(img, sigma1, sigma2):
    n, m = img.shape

    #crating the tow 1D kernels
    gauss_row = np.zeros((n, 1), dtype = float)
    gauss_col = np.zeros((1, m), dtype = float)

    #calculating the kernels' center
    a = int((n-1)/2)
    b = int((m-1)/2)

    #calculating the values
    for i in range(0, n):
        gauss_row[i,0] = gaussian_kernel(sigma1, i-a)

    for i in range(0, m):
        gauss_col[0,i] = gaussian_kernel(sigma2, i-b)

    #multiplying the tow arrays to creatr a gaussian matrix
    f = np.matmul(gauss_row , gauss_col)

    #multiplying the original image by the filter
    img_out = np.multiply(img, f)

    return img_out

def spatial_gaussian_component(sigma,size):
    filter = np.zeros((size,size))
    a = int(size/2)
    for i in range(0,size):
        for j in range(0,size):
            filter[i][j] = gaussian_kernel(sigma,euclidian_distance(i-a,j-a,0,0))
    return filter


def range_gaussian_component(img,sigma,size,x,y):
    range_gaussian = np.zeros((size,size))
    center_intensity = img[x][y]
    a = int((size-1)/2)#kernel center
    neighbor_intensities = img[x-a:x+a+1,y-a:y+a+1]
    range_gaussian = gaussian_kernel(sigma,neighbor_intensities-center_intensity)
    return range_gaussian

def convolution(img, f):
    #output image
    out_img = np.zeros(img.shape)

    #fliping the filter in
    f_flip = np.flip(np.flip(f,0),1)

    #getting the dimensios of the filter
    f_n,f_m= f.shape

    fa = int((f_n-1)/2)
    fb = int((f_m-1)/2)

    #padding the image and getting its dimensions
    pd_img = padding_image(img, f_n)
    pd_img_n,pd_img_m = pd_img.shape

    #applying the convolution
    for x in range(fa, (pd_img_n-fa)):
        for y in range(fb, (pd_img_m - fb)):

            img_region = pd_img[ x-fa:x+(fa+1), y-fb:y+(fb+1)]

            out_img[(x-fa),(y-fb)] = np.sum(np.multiply(img_region, f_flip))

    return out_img

def bilateral_filter(input_img, spatial_gaussian,sigma_r):

    input_img = input_img.astype(np.float64)

    #output image
    out_img = np.zeros(input_img.shape)


    #getting the dimensios of the filter
    filter_size = spatial_gaussian.shape[0]

    fa = int((filter_size-1)/2)

    #padding the image and getting its dimensions
    pd_img = padding_image(input_img, filter_size)
    pd_img_n,pd_img_m = pd_img.shape


    #applies the convolution
    for x in range(fa, (pd_img_n-fa)):
        for y in range(fa, (pd_img_m - fa)):
            final_intensity, wp = [0,0]
            range_gaussian = range_gaussian_component(pd_img,sigma_r,filter_size,x,y)
            w_filter = spatial_gaussian * range_gaussian
            wp = np.sum(w_filter)
            img_region = pd_img[ x-fa:x+(fa+1), y-fa:y+(fa+1)]
            final_intensity = np.sum(np.multiply(img_region, w_filter))
            out_img[(x-fa),(y-fa)] = final_intensity/wp
    return out_img


def normalization(img):
    m = img.min()
    M = img.max()
    return (((img-m)*255.0)/(M-m))



#this functions gets the parameters for each method and calls for the tranformation itself
def apply_bilateral_filter(input_img):
    #input
    filter_size = int(input())
    sigma_s = float(input())
    sigma_r = float(input())
    spatial_gaussian = spatial_gaussian_component(sigma_s,filter_size)
    return bilateral_filter(input_img,spatial_gaussian,sigma_r)

def apply_laplacian_filter(input_img):
    #kernels
    kernel1 = np.matrix([[0,-1,0], [-1,4,-1], [0,-1,0]])
    kernel2 = np.matrix([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])

    #input
    c = float(input())
    kn_option = int(input())

    #kernel choice
    if(kn_option == 1):
        img_aux = convolution(input_img, kernel1) #convolving
    else:
        img_aux = convolution(input_img, kernel2) #convolving

    img_aux = normalization(img_aux) #saling
    img_aux = (c * img_aux) + input_img #adding
    img_out = normalization(img_aux) #scaling


    return img_out

def apply_vignette_filter(input_img):
    sigma1 = float(input())
    sigma2 = float(input())
    #applying filter
    img_aux = vignette(input_img,sigma1,sigma2)
    #scaling
    img_out = normalization(img_aux)

    return img_out

#puts the function in an array, so i can call them only once without if-else structures
methods = [apply_bilateral_filter, apply_laplacian_filter, apply_vignette_filter]



#reads the image
filename = str(input()).rstrip()
input_img = im.imread(filename)

method = int(input())
save_option = int(input())

#apply the correct filter to the image
transformed_image = (methods[method-1])(input_img)

#rse print
print("%.4f" % (rse(input_img,transformed_image)))

#save option
if save_option == 1:
    im.imwrite('output_img.png',transformed_image.astype(np.uint8))
