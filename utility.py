
import os
import pandas as pd

import imageio
import nibabel as nib
import scipy.ndimage as ndi
import cv2
import SimpleITK as sitk
from skimage import io
from skimage.measure import find_contours

import math
import numpy as np
import cv2
import SimpleITK as sitk
from skimage import io
from skimage.measure import find_contours

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt




def get_bounding_box(mask, crop_margin=0):
    """
    Return the bounding box of a mask image.
    slightly modify from https://github.com/guillaumefrd/brain-tumor-mri-dataset/blob/master/data_visualization.ipynb
    """
    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    for row in range(mask.shape[0]):
        if mask[row, :].max() != 0:
            ymin = row + crop_margin
            break

    for row in range(mask.shape[0] - 1, -1, -1):
        if mask[row, :].max() != 0:
            ymax = row + crop_margin
            break

    for col in range(mask.shape[1]):
        if mask[:, col].max() != 0:
            xmin = col + crop_margin
            break

    for col in range(mask.shape[1] - 1, -1, -1):
        if mask[:, col].max() != 0:
            xmax = col + crop_margin
            break

    return xmin, ymin, xmax, ymax


def crop_to_bbox(image, bbox, crop_margin=0):
    """
    Crop an image to the bounding by forcing a squared image as output.
    from https://github.com/guillaumefrd/brain-tumor-mri-dataset/blob/master/data_visualization.ipynb
    """
    x1, y1, x2, y2 =  bbox

    # force a squared image
    max_width_height = np.maximum(y2 - y1, x2 - x1)
    y2 = y1 + max_width_height
    x2 = x1 + max_width_height

    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)
    y2 = np.minimum(y2 + crop_margin, image.shape[0])
    x1 = np.maximum(x1 - crop_margin, 0)
    x2 = np.minimum(x2 + crop_margin, image.shape[1])

    return image[y1:y2, x1:x2]


def has_tumor(mask):
    '''
    mask should be a binary 2D array (0 : pixel do not contained tumor , 1 : pixel containing tumor)
    Return True if any pixel values of mask are equal to 1
    '''
    return sum(mask.ravel()) > 0

def get_index_tumor_slice(masks):
  '''
    Return index of images/masks where tumor is visible
    Masks are 2D or 3D array with same dimensions
    Masks should contain at least one slice with tumor
  '''
  tumor_idx = []
  if len(masks.shape) == 3:
    for i in range(0, masks.shape[2]):
      # detect binary mask with no tumor (all pixel equal to 0)
      if has_tumor(masks[:,:,i]):
        tumor_idx.append(i)
  elif len(masks.shape) == 2:
      # detect binary mask with no tumor (all pixel equal to 0)
    if has_tumor(masks[:,:,i]):
      tumor_idx.append(i)

  if not tumor_idx:
    raise ValueError("There must be at least one slice with tumor")
  else:
    return tumor_idx



def image_picker(min_slice, max_slice):
    '''
    return an array containing (max 18) equidistant index number between min_slice and max_slice
    min_slice and max_slice are interger (min_slice cannot be >= max_slice)
    '''
    row = 3
    col = 6
    picker = math.ceil((max_slice - min_slice)/(row*col))
    index_list = []
    index = min_slice

    for i in range(row*col):
      if index < max_slice:
        index_list.append(index)
        index += picker
      else:
        index_list.append(max_slice)
        break
    return index_list



def plot_bbox_image(images, masks, crop_margin=0, zooming=False):
  '''
    Plot equidistant slices with bounding boxe containing the tumor
    masks is binary (0 : pixel do not contained tumor , 1 : pixel containing tumor)
    images and masks are 2D array with same dimensions
  '''

  if len(masks.shape) != 2:
    raise ValueError("only accept one array of 2D dimension")

  xmin, ymin, xmax, ymax = get_bounding_box(masks, crop_margin)

  plt.imshow(images, cmap='gray')
  plt.plot([xmin, xmax], [ymin, ymin], color='red')
  plt.plot([xmax, xmax], [ymin, ymax], color='red')
  plt.plot([xmin, xmin], [ymin, ymax], color='red')
  plt.plot([xmin, xmax], [ymax, ymax], color='red')

  if zooming:
    plt.plot([xmax, 511], [ymax, 511], color='red')
    plt.plot([xmax, 511], [ymin, 0], color='red')



def PlotImage(images, masks=None, show_tumor_only = False, dislay_mode=None):
  '''
    Plot equidistant slices along mask and corresponding images along col/row grid
    masks is binary (0 : pixel do not contained tumor , 1 : pixel containing tumor)
    images and masks are 2D or 3D array with same dimensions
    if masks = None, Only images are displayed
    if show_tumor_only = True: Only images containing tumor and/or corresponding masks are displayed

    if dislay_mode = 'mask' : Display tumor mask with corresponding images
    if dislay_mode = 'frame' : Display bounding box with corresponding images
    if dislay_mode = 'contour' : Display tumor contour with corresponding images
  '''
  row = 3
  col = 6
  images = np.rot90(images, axes=(1, 0))

  if show_tumor_only:
    images_idx = get_index_tumor_slice(masks)
    slices_picker = image_picker(min_slice = images_idx[0], max_slice=images_idx[-1])
  else:
    slices_picker = image_picker(min_slice = 0, max_slice=images.shape[2]-1)

  if masks is not None:
    masks = np.rot90(masks, axes=(1, 0))

  plt.figure(figsize=(16, 8))
  for i, idx in enumerate(slices_picker):
    plt.subplot(row, col, i+1)
    plt.imshow(images[:,:,idx], cmap='gray')

    if masks is not None:
      if dislay_mode == 'mask':
        tumor = np.ma.masked_where(masks[:,:,idx] == False, masks[:,:,idx])
        plt.imshow(tumor, cmap='Set1')

      if dislay_mode == 'frame':
        plot_bbox_image(images[:,:,idx], masks[:,:,idx], crop_margin=0, zooming=False)

      if dislay_mode == 'contour':
        contours = find_contours(masks[:,:,idx],0)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, c='r')
    plt.title(str(idx+1))
    plt.axis('off')
  plt.subplots_adjust(wspace=0.05, hspace=0.2)
  plt.show()

