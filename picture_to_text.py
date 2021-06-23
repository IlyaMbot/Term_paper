import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import glob
import os

def get_points(image):
    LX = len(image[0])
    LY = len(image)
    data = np.zeros(LX)
    image = np.array(image)
    #print(image[:,50])

    for i in range(LX):
        pixel_position = ndimage.measurements.center_of_mass(image[:,i])
        if(pixel_position[0] > 0): 
            data[i] = np.int32(pixel_position[0])

    return(-1 * data[5:-5] + LY )

def general_plot(image, data, index):
    lsize = 16

    fig, ax = plt.subplots(2, 1)
    
  
    ax[0].imshow(image)
    ax[0].set_title("image", size = lsize)

    ax[1].plot(data, color = "black" )
    ax[1].set_title("Result", size = lsize)
        
    plt.tight_layout()
    plt.savefig('data/Aschw_data/digitpic{}.png'.format(index), transparent=False, dpi=500, bbox_inches="tight")
    #plt.show()

def save_dat(data, index):
    filepath = "data/Aschw_data/digtext{}.dat".format(index)
    
    length = len(data)
    time = np.arange(0, length)

    with open(filepath, "w") as file2w:
        for i in range(length):
            file2w.write("{}\t{}\n".format(time[i], data[i]))

filenames = glob.glob("data/Aschw_data/*.png")

filenames = sorted(filenames, key=os.path.basename)

number = 0
for filename in filenames:
    number +=1
    print(filename)
    img = mpimg.imread(filename)
    image = []
    
    
    for i in range(len(img[:,0,0])):
        y = []
        for j in range(len(img[0,:,0])):
            pixel = 0
            for k in range(len(img[0,0,:])):
                pixel += img[i,j,k]
            y.append(pixel - 1)
        image.append(y)

    data = get_points(image)
    #general_plot(image, data, number)
    save_dat(data, number)