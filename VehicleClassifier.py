import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
import random
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class VehicleClassifier:

    def __init__(self,
                 color_space='RGB',
                 spatial_size=(32, 32),
                 color_histogram_bins=32,
                 orientations=9,
                 pixels_per_cell=8,
                 cells_per_block=2,
                 hog_channels=[0, 1, 2],
                 spatial_features=True,
                 color_histogram_features=True,
                 hog_features=True):
        self._color_space = color_space
        self._spatial_size = spatial_size
        self._color_histogram_bins = color_histogram_bins
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
        self._hog_channels = hog_channels
        self._spatial_features = spatial_features
        self._color_histogram_features = color_histogram_features
        self._hog_features = hog_features
        self._classifier = None
        self._scaler = None

    def train(self,
              vehicles_paths,
              non_vehicles_paths,
              sample_size=None):

        cars = []
        for path in vehicles_paths:
            cars += glob.glob(path)
        not_cars = []
        for path in non_vehicles_paths:
            not_cars += glob.glob(path)

        print('Found {0} car samples'.format(len(cars)))
        print('Found {0} non-car samples'.format(len(not_cars)))
        if sample_size is None:
            sample_size = min(len(cars), len(not_cars))

        random.seed(0)
        random.shuffle(cars)
        cars = cars[0:sample_size]
        random.shuffle(not_cars)
        not_cars = not_cars[0:sample_size]

        features = []
        for file in cars + not_cars:
            image = cv2.imread(file)
            features.append(self._extract_features(image, (0, 256)))
        print('Feature vector length:', len(features[0]))
        print('Total samples: ', len(features))

        x = np.array(features).astype(np.float64)
        self._scaler = StandardScaler().fit(x)
        scaled_x = self._scaler.transform(x)
        y = np.concatenate([np.ones(len(cars)), np.zeros(len(not_cars))])

        x_train, x_test, y_train, y_test = \
            train_test_split(scaled_x, y, test_size=0.2, random_state=27)

        self._classifier = LinearSVC()
        t = time.time()
        self._classifier.fit(x_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train classifier...')
        print('Test Accuracy of classifier = ', round(self._classifier.score(x_test, y_test), 4))

    def search_vehicles(self,
                        img,
                        heatmap,
                        windows):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self._extract_features(test_img, histogram_range=(0, 256))
            test_features = self._scaler.transform(np.array(features).reshape(1, -1))
            prediction = self._classifier.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
                heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
        return on_windows

    def fast_search_vehicles(self,
                             img,
                             heatmap,
                             x_start_stop,
                             y_start_stop,
                             scale=1,
                             cells_per_step=2):
        on_windows = []

        feature_image = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]
        feature_image = self._convert_color(feature_image)
        if scale != 1:
            feature_image = cv2.resize(feature_image, (np.int(feature_image.shape[1] / scale),
                                                       np.int(feature_image.shape[0] / scale)))

        img_x_size = feature_image.shape[1]
        img_y_size = feature_image.shape[0]
        x_blocks = (img_x_size // self._pixels_per_cell) - 1
        y_blocks = (img_y_size // self._pixels_per_cell) - 1
        # todo: 64 shouldn't be hardcoded - it's the size of the training images?
        window_size = 64
        blocks_per_window = (window_size // self._pixels_per_cell) - 1
        x_steps = (x_blocks - blocks_per_window) // cells_per_step
        y_steps = (y_blocks - blocks_per_window) // cells_per_step

        img_hog_features = {}
        if self._hog_features:
            for channel in self._hog_channels:
                img_hog_features[channel] = self._get_hog_features(feature_image[:, :, channel],
                                                                   visualize=False,
                                                                   feature_vector=False)
        window_count = 0
        for x in range(x_steps):
            for y in range(y_steps):
                window_count += 1

                features = []
                x_pos = x * cells_per_step
                y_pos = y * cells_per_step
                x_left = x_pos * self._pixels_per_cell
                y_top = y_pos * self._pixels_per_cell

                # todo: 64 shouldn't be hardcoded - it's the size of the training images?
                img_window = cv2.resize(feature_image[y_top:y_top + window_size, x_left:x_left + window_size],
                                        (64, 64))
                if self._spatial_features:
                    feat = self._bin_spatial(img_window)
                    features.append(feat)

                if self._color_histogram_features:
                    feat = self._color_hist(img_window, bins_range=(0, 256))
                    features.append(feat)

                if self._hog_features:
                    feat = []
                    for channel in self._hog_channels:
                        channel_hog = img_hog_features[channel][y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel()
                        feat.append(channel_hog)
                    features.append(np.concatenate(feat))

                test_features = self._scaler.transform(np.concatenate(features).reshape(1, -1))
                prediction = self._classifier.predict(test_features)
                if prediction == 1:
                    box_left = np.int(x_left * scale)
                    box_top = np.int(y_top * scale)
                    box_window = np.int(window_size * scale)
                    box_ll = (box_left + x_start_stop[0], box_top + y_start_stop[0])
                    box_ur = (box_ll[0] + box_window, box_ll[1] + box_window)
                    on_windows.append((box_ll, box_ur))
                    heatmap[box_ll[1]:box_ur[1], box_ll[0]:box_ur[0]] += 1
        return on_windows

    def _convert_color(self, image):
        if self._color_space != 'RGB':
            if self._color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif self._color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif self._color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif self._color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif self._color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = np.copy(image)
        return feature_image

    def _extract_features(self,
                          image,
                          histogram_range):
        features = []
        feature_image = self._convert_color(image)

        if self._spatial_features:
            feat = self._bin_spatial(feature_image)
            features.append(feat)

        if self._color_histogram_features:
            feat = self._color_hist(feature_image, bins_range=histogram_range)
            features.append(feat)

        if self._hog_features:
            hog_features = []
            for channel in self._hog_channels:
                hog_features.append(self._get_hog_features(feature_image[:, :, channel],
                                                           visualize=False,
                                                           feature_vector=True))
            hog_features = np.ravel(hog_features)
            features.append(hog_features)

        return np.concatenate(features)

    def _get_hog_features(self,
                          img,
                          visualize=False,
                          feature_vector=True):
        """
        Extracts HOG features from a channel
        If visualize is True, also returns an image representing the extracted HOG features
        """
        if visualize:
            features, hog_image = hog(img,
                                      orientations=self._orientations,
                                      pixels_per_cell=(self._pixels_per_cell, self._pixels_per_cell),
                                      cells_per_block=(self._cells_per_block, self._cells_per_block),
                                      transform_sqrt=False,
                                      visualise=visualize,
                                      feature_vector=feature_vector)
            return features, hog_image
        else:
            features = hog(img,
                           orientations=self._orientations,
                           pixels_per_cell=(self._pixels_per_cell, self._pixels_per_cell),
                           cells_per_block=(self._cells_per_block, self._cells_per_block),
                           transform_sqrt=False,
                           visualise=visualize,
                           feature_vector=feature_vector)
            return features

    def _bin_spatial(self, img):
        """
        Resizes an image and flattens the result in a 1-D array
        """
        features = cv2.resize(img, self._spatial_size).ravel()
        return features

    def _color_hist(self, img, bins_range=(0, 256)):
        """
        Computes the histogram for each image channel and concatenates it into a single array
        Expects a 3-channel image
        """
        channel1_hist = np.histogram(img[:, :, 0], bins=self._color_histogram_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self._color_histogram_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self._color_histogram_bins, range=bins_range)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features
