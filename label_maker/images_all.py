# pylint: disable=unused-argument
"""Generate an .npz file containing arrays for training machine learning algorithms"""

from os import makedirs, path as op
from random import shuffle

import numpy as np
import os
from label_maker.utils import download_tile_tms_all, get_tile_tif

def download_images_all(dest_folder, classes, imagery, ml_type, background_ratio, imagery_offset=False, **kwargs):
    """Download satellite images specified by a URL and a label.npz file
    Parameters
    ------------
    dest_folder: str
        Folder to save labels, tiles, and final numpy arrays into
    classes: list
        A list of classes for machine learning training. Each class is defined as a dict
        with two required properties:
          - name: class name
          - filter: A Mapbox GL Filter.
        See the README for more details
    imagery: str
        Imagery template to download satellite images from.
        Ex: http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN
    ml_type: str
        Defines the type of machine learning. One of "classification", "object-detection", or "segmentation"
    background_ratio: float
        Determines the number of background images to download in single class problems. Ex. A value
        of 1 will download an equal number of background images to class images.
    imagery_offset: list
        An optional list of integers representing the number of pixels to offset imagery. Ex. [15, -5] will
        move the images 15 pixels right and 5 pixels up relative to the requested tile bounds
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """
    # open labels file
    labels_file = op.join(dest_folder, 'labels.npz')
    tiles = np.load(labels_file)

    # create tiles directory
    tiles_dir = op.join(dest_folder, 'tiles_all')
    if not op.isdir(tiles_dir):
        makedirs(tiles_dir)

    tiles  = [tile for tile in tiles.files ]
	
    print('Downloading all {} tiles to {}'.format(len(tiles), op.join(dest_folder, 'tiles_all')))

    # get image acquisition function based on imagery string
    image_function = download_tile_tms_all
    if op.splitext(imagery)[1].lower() in ['.tif', '.tiff']:
        image_function = get_tile_tif  
		
    for tile in tiles:
        image_function(tile, imagery, dest_folder, imagery_offset)
         
