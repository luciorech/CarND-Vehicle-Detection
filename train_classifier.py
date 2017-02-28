import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import argparse
import pickle

from VehicleClassifier import VehicleClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driving Lane Detection')
    parser.add_argument('--vehicles', default='./datasets/vehicles/*/*.png')
    parser.add_argument('--non-vehicles', default='./datasets/non-vehicles/*/*.png')
    parser.add_argument('--color-space', default='RGB')
    parser.add_argument('--sample-size', default=None, type=int)
    parser.add_argument('--filename', default='clf.p')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)
    # Spatial features
    parser.add_argument('--spatial-features', dest='spatial_features', action='store_true')
    parser.add_argument('--no-spatial-features', dest='spatial_features', action='store_false')
    parser.set_defaults(spatial_features=True)
    parser.add_argument('--spatial-size', default=32, type=int)
    # Color histogram features
    parser.add_argument('--color-hist-features', dest='color_hist_features', action='store_true')
    parser.add_argument('--no-color-hist-features', dest='color_hist_features', action='store_false')
    parser.set_defaults(color_hist_features=True)
    parser.add_argument('--color-hist-bins', default=32, type=int)
    # HOG features
    parser.add_argument('--hog-features', dest='hog_features', action='store_true')
    parser.add_argument('--no-hog-features', dest='hog_features', action='store_false')
    parser.set_defaults(hog_features=True)
    parser.add_argument('--hog-orientations', default=9, type=int)
    parser.add_argument('--hog-pixels-per-cell', default=8, type=int)
    parser.add_argument('--hog-cells-per-block', default=2, type=int)
    parser.add_argument('--hog-channels', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()

    vehicles_path = [args.vehicles]
    non_vehicles_path = [args.non_vehicles]

    clf = VehicleClassifier(color_space=args.color_space,
                            spatial_size=(args.spatial_size, args.spatial_size),
                            color_histogram_bins=args.color_hist_bins,
                            orientations=args.hog_orientations,
                            pixels_per_cell=args.hog_pixels_per_cell,
                            cells_per_block=args.hog_cells_per_block,
                            hog_channels=args.hog_channels,
                            spatial_features=args.spatial_features,
                            color_histogram_features=args.color_hist_features,
                            hog_features=args.hog_features)
    clf.train(vehicles_path, non_vehicles_path, sample_size=args.sample_size)

    with open(args.filename, 'wb') as outfile:
        pickle.dump(clf, outfile)

    if args.debug:
        image = cv2.imread('./datasets/vehicles/KITTI_extracted/1.png')
        print(image.shape)
        color_cvt = clf.convert_color(image)
        print(color_cvt.shape)
        _, hog_img_0 = clf.get_hog_features(color_cvt[:, :, 0], visualize=True)
        _, hog_img_1 = clf.get_hog_features(color_cvt[:, :, 1], visualize=True)
        _, hog_img_2 = clf.get_hog_features(color_cvt[:, :, 2], visualize=True)
        f, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0][0].imshow(image[..., ::-1])
        ax[0][1].imshow(hog_img_0, cmap="gray")
        ax[1][0].imshow(hog_img_1, cmap="gray")
        ax[1][1].imshow(hog_img_2, cmap="gray")
        plt.show(block=True)

        for img_path in glob.glob('./test_images/*.jpg'):
            image = cv2.imread(img_path)
            heatmap = np.zeros_like(image[:, :, 0])
            hot_windows = clf.fast_search_vehicles(image,
                                                   heatmap,
                                                   x_start_stop=[0, 1279],
                                                   y_start_stop=[400, 719],
                                                   scale=1,
                                                   cells_per_step=2)
            draw_image = np.copy(image)
            for bbox in hot_windows:
                cv2.rectangle(draw_image, bbox[0], bbox[1], (0, 255, 0), 6)

            f, ax = plt.subplots(1, 2, figsize=(18, 18))
            ax[0].imshow(draw_image[...,::-1])
            ax[1].imshow(heatmap, cmap="gray")
            plt.show(block=True)
