import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import image_processing as ip
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import pickle
import matplotlib.image as mpimg

from VehicleClassifier import VehicleClassifier

if __name__ == "__main__":
    vehicles_path = './datasets/vehicles/GTI_Far/*.png'
    non_vehicles_path = './datasets/non-vehicles/GTI/*.png'

    color_space = 'YUV'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"
    spatial_feat = False
    hist_feat = False
    hog_feat = True

    v_classifier = VehicleClassifier(color_space=color_space,
                                     spatial_size=spatial_size,
                                     color_histogram_bins=hist_bins,
                                     orientations=orient,
                                     pixels_per_cell=pix_per_cell,
                                     cells_per_block=cell_per_block,
                                     hog_channels=[0, 1, 2],
                                     spatial_features=spatial_feat,
                                     color_histogram_features=hist_feat,
                                     hog_features=hog_feat)
    v_classifier.train(vehicles_path, non_vehicles_path, 500)

    # Check the prediction time for a single sample
    t = time.time()

    image = mpimg.imread('./test_images/test3.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    windows = ip.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                              xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = ip.search_windows(image, windows, v_classifier._classifier, v_classifier._scaler,
                                    color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)

    window_img = ip.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.figure(figsize=(24, 18))
    plt.imshow(window_img)
    plt.show(block=True)
