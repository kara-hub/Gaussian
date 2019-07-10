from __future__ import print_function
import gaussian_random_fields as gr
from scipy.ndimage import median_filter
from scipy.signal import convolve2d 
import numpy as np
import scipy.ndimage as ndimage
import numpy as np
import os


def split_overlap(array,size,overlap):  #for 2D
    result = []
    result_split = []

    while True:
        if len(array) <= size:
            result.append(array)
            break
        else:
            result.append(array[:size])
            array = array[size-overlap:]
            
    for block in result:
        while True:
            if len(block[0]) <= size:
                result_split.append(block)
                break
            else:
                result_split.append(block[:size,:size])
                block = block[:size, size-overlap:]
            
    return result_split


def gen_dataset(num_fields, img_size=1024, slice_dim=128):
    
    os.makedirs('data', exist_ok=True)

    for n in range(1, num_fields + 1):
        field  = gr.gaussian_random_field(alpha = 1, size = img_size)
        avg = np.mean(field)
        field[field > avg/2] = 1
        img = ndimage.gaussian_filter(field, sigma=(10, 10), order=0)
        avg = np.mean(img)
        img[img > avg*1.01] = 1 # not imporants just smoothes the edges a little bit
        img[img < avg/1.01] = 0 # not imporants just smoothes the edges a little bit

        # save field matrix
        np.save('data/field_%d.npy' % n, img)
        
        #overlap set to 50%
        overlap = slice_dim // 2
        
        #type of data generated (overlap or no overlap, generates both as is)
        #creates 128*128 blocks with overlap (50%)
        new_split = np.array(split_overlap(img, slice_dim, overlap))
        #creates unique 128*128 blocks (no overlap)
        no_ovr_split = np.array(split_overlap(img, slice_dim, 0))

        block_ct = 0
        for arr in new_split:
            block_ct += 1
            np.save('data/real_ovr_%d_%d.npy' % (n, block_ct), arr)
            
            fourier_block = np.abs(np.fft.fftshift(np.fft.fft2(arr)))**2
            fourier_block[(len(fourier_block) // 2) - 1][(len(fourier_block) // 2) - 1] = 1
            np.save('data/fft_ovr_%d_%d.npy'% (n, block_ct), fourier_block)
            
            #print(arr.shape)
        
        for arr in no_ovr_split:
            block_ct += 1
            np.save('data/real_no_%d_%d.npy' % (n, block_ct), arr)
            
            fourier_block = np.abs(np.fft.fftshift(np.fft.fft2(arr)))**2
            
            #change to //2, no -1 [64][64], not [63][63]
            fourier_block[(len(fourier_block) // 2) - 1][(len(fourier_block) // 2) - 1] = 1
            np.save('data/fft_no_%d_%d.npy'% (n, block_ct), fourier_block)
            
            #print(arr.shape)



# gen_dataset(2)





