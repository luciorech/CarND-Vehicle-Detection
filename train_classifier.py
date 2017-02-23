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
    vehicles_path = ['./datasets/vehicles/*/*.png']
    non_vehicles_path = ['./datasets/non-vehicles/*/*.png']

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

    clf_1 = VehicleClassifier(color_space=color_space,
                              spatial_size=spatial_size,
                              color_histogram_bins=hist_bins,
                              orientations=orient,
                              pixels_per_cell=pix_per_cell,
                              cells_per_block=cell_per_block,
                              hog_channels=[0, 1, 2],
                              spatial_features=spatial_feat,
                              color_histogram_features=hist_feat,
                              hog_features=hog_feat)
    clf_1.train(vehicles_path, non_vehicles_path, sample_size=400)

    clf_2 = VehicleClassifier(color_space='YCrCb',
                              spatial_size=spatial_size,
                              color_histogram_bins=hist_bins,
                              orientations=orient,
                              pixels_per_cell=pix_per_cell,
                              cells_per_block=cell_per_block,
                              hog_channels=[0, 1, 2],
                              spatial_features=False,
                              color_histogram_features=False,
                              hog_features=hog_feat)
    clf_2.train(vehicles_path, non_vehicles_path, sample_size=400)

    # Check the prediction time for a single sample
    t = time.time()

    for img_path in glob.glob('./test_images/*.jpg'):
        image = cv2.imread(img_path)
        draw_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # image = image.astype(np.float32)/255
        heatmap = np.zeros_like(image[:, :, 0])

        windows = []
        windows += ip.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                                  xy_window=(64, 64), xy_overlap=(0.5, 0.5))
        windows += ip.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                                  xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        windows += ip.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                                  xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        print("Total search windows = {0}".format(len(windows)))
        hot_windows = clf_1.search_vehicles(image, heatmap, windows, scale=(64, 64))
        hot_windows += clf_2.search_vehicles(image, heatmap, windows, scale=(64, 64))

        window_img = ip.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        f, ax = plt.subplots(1, 2, figsize=(48, 18))
        ax[0].imshow(window_img[...,::-1])
        ax[1].imshow(heatmap, cmap="gray")
        plt.show(block=True)
