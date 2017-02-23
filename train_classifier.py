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

if __name__ == "__main__":
    vehicles_path = './datasets/vehicles/GTI_Far/*.png'
    non_vehicles_path = './datasets/non-vehicles/GTI/*.png'

    cars = glob.glob(vehicles_path)
    not_cars = glob.glob(non_vehicles_path)

    sample_size = 500

    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    color_space = 'YUV'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    car_features = ip.extract_features_files(cars,
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
    not_car_features = ip.extract_features_files(not_cars,
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

    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

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

    hot_windows = ip.search_windows(image, windows, svc, X_scaler,
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
