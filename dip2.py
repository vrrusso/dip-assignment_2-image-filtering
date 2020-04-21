# Assignment 2 : Image Enhancement and Filtering
# Authors      : Paulo Rocha nusp
#                Victor R. Russo 11218855
# Course       : scc025 - Digital Image Processing
# Period       : 2020 First Semester
#

import numpy as np
import imageio as im



filename = str(input()).rstrip()
input_img = im.imread(filename)

method = int(input())
save_option = int(input())


