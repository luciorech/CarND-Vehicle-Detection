import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


def get_hog_features(img,
                     orient,
                     pix_per_cell,
                     cell_per_block,
                     visualize=False,
                     feature_vector=True):
    """
    Extracts HOG features from a channel
    If visualize is True, also returns an image representing the extracted HOG features
    """
    if visualize:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=visualize,
                                  feature_vector=feature_vector)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=visualize,
                       feature_vector=feature_vector)
        return features


def bin_spatial(img, size=(32, 32)):
    """
    Resizes an image and flattens the result in a 1-D array
    """
    features = cv2.resize(img, size).ravel()
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, bins=32, bins_range=(0, 256)):
    """
    Computes the histogram for each image channel and concatenates it in a single array
    """
    # todo: Expects a 3-channel image, make this generic
    channel1_hist = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=bins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features(image,
                     color_space='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_features=True,
                     color_histogram_features=True,
                     hog_features=True):
    features = []
    # todo: improve conversion, deal with BGR inputs
    # todo: do we really need to copy the image?
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    if spatial_features:
        feat = bin_spatial(feature_image, size=spatial_size)
        features.append(feat)

    if color_histogram_features:
        feat = color_hist(feature_image, bins=hist_bins)
        features.append(feat)

    if hog_features:
        if hog_channel == 'ALL':
            hog_channels = list(range(feature_image.shape[2]))
        else:
            hog_channels = [hog_channel]
        hog_features = []
        for channel in hog_channels:
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient,
                                                 pix_per_cell,
                                                 cell_per_block,
                                                 visualize=False,
                                                 feature_vector=True))
        hog_features = np.ravel(hog_features)
        features.append(hog_features)

    return np.concatenate(features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_files(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = extract_features(image,
                                         color_space=color_space,
                                         spatial_size=spatial_size,
                                         hist_bins=hist_bins,
                                         orient=orient,
                                         pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel,
                                         spatial_features=spatial_feat,
                                         color_histogram_features=hist_feat,
                                         hog_features=hog_feat)
        features.append(file_features)
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - nx_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = extract_features(test_img, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_features=spatial_feat,
                                    color_histogram_features=hist_feat, hog_features=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
