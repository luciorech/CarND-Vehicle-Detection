from VehicleDetection import VehicleDetection
from VehicleClassifier import VehicleClassifier
from moviepy.editor import VideoFileClip
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driving Lane Detection')
    parser.add_argument('--input', default='./project_video.mp4', help='Path to input video file.')
    parser.add_argument('--output', default='./output_videos/project_video.mp4', help='Path to output video file.')
    parser.add_argument('--start', default=0, type=float)
    parser.add_argument('--end', default=None, type=float)
    parser.add_argument('--sample_size', default=None, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    vehicles_path = ['./datasets/vehicles/*/*.png']
    non_vehicles_path = ['./datasets/non-vehicles/*/*.png']

    clf_1 = VehicleClassifier(color_space='YUV',
                              spatial_size=(32, 32),
                              color_histogram_bins=32,
                              orientations=9,
                              pixels_per_cell=8,
                              cells_per_block=2,
                              hog_channels=[0],
                              spatial_features=True,
                              color_histogram_features=True,
                              hog_features=True)
    clf_1.train(vehicles_path, non_vehicles_path, sample_size=args.sample_size)

    clf_2 = VehicleClassifier(color_space='YCrCb',
                              spatial_size=(32, 32),
                              color_histogram_bins=32,
                              orientations=9,
                              pixels_per_cell=8,
                              cells_per_block=2,
                              hog_channels=[0, 1, 2],
                              spatial_features=False,
                              color_histogram_features=False,
                              hog_features=True)
    clf_2.train(vehicles_path, non_vehicles_path, sample_size=args.sample_size)

    v_detection = VehicleDetection([clf_1, clf_2], debug_mode=args.debug)
    # v_detection = VehicleDetection([clf_1], debug_mode=args.debug)
    v_detection.create_sliding_windows(x_start_stop=[0, 1279], y_start_stop=[400, 719])
    start = float(args.start)
    end = float(args.end) if args.end is not None else None
    in_video = VideoFileClip(args.input).subclip(t_start=start, t_end=end)
    out_video = in_video.fl_image(v_detection.process_frame)
    out_video.write_videofile(args.output, audio=False)
