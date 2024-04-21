import numpy as np
from skimage import io as io_url
import matplotlib.pyplot as plt
import cv2
import math


pi = math.pi
def DFT_slow(data):
  """
  Implement the discrete Fourier Transform for a 1D signal
  params:
    data: Nx1: (N, ): 1D numpy array
  returns:
    DFT: Nx1: 1D numpy array 
  """
  N = data.shape[0]
  j = complex('j')
  A = np.array([[math.e ** (-j*2*pi*s*n/N) for n in range(N)] for s in range(N)])
  print(A.shape)
  ret = A@data.reshape(N, 1)
  # assert abs((ret - np.fft.fft(data)).mean()) < 1e-8
  return ret.reshape(N,)


def show_img(origin, row_fft, row_col_fft):
    """
    Show the original image, row-wise FFT and column-wise FFT

    params:
        origin: (H, W): 2D numpy array
        row_fft: (H, W): 2D numpy array
        row_col_fft: (H, W): 2D numpy array    
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    axs[0].imshow(origin, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.log(np.abs(np.fft.fftshift(row_fft))), cmap='gray')
    axs[1].set_title('Row-wise FFT')
    axs[1].axis('off')
    axs[2].imshow((np.log(np.abs(np.fft.fftshift(row_col_fft)))), cmap='gray')
    axs[2].set_title('Column-wise FFT')
    axs[2].axis('off')
    plt.show()


def DFT_2D(gray_img):
    """
    Implement the 2D Discrete Fourier Transform
    Note that: dtype of the output should be complex_
    params:
        gray_img: (H, W): 2D numpy array
        
    returns:
        row_fft: (H, W): 2D numpy array that contains the row-wise FFT of the input image
        row_col_fft: (H, W): 2D numpy array that contains the column-wise FFT of the input image
    """
    R, C = gray_img.shape
    row_fft = np.zeros_like(gray_img)
    for r in range(R):
      row_fft[r] = np.fft.fft(gray_img[r])
    row_col_fft = row_fft
    for c in range(C):
      row_col_fft[:, c] = np.fft.fft(row_fft[:, c])
    
    # assert abs((row_col_fft - np.fft.fft2(gray_img)).mean()) < 1e-8
    
    return row_fft, row_col_fft



if __name__ == '__main__':
  
    # ex1
    # x = np.random.random(1024)
    # print(np.allclose(DFT_slow(x), np.fft.fft(x)))
  # ex2
    img = io_url.imread('https://img2.zergnet.com/2309662_300.jpg')
    gray_img = np.mean(img, -1)
    row_fft, row_col_fft = DFT_2D(gray_img)
    show_img(gray_img, row_fft, row_col_fft)

 



