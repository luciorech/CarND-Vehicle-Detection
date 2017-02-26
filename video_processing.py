from VehicleDetection import VehicleDetection
from moviepy.editor import VideoFileClip
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driving Lane Detection')
    parser.add_argument('--input', default='./project_video.mp4', help='Path to input video file.')
    parser.add_argument('--output', default='./output_videos/project_video.mp4', help='Path to output video file.')
    parser.add_argument('--heatmap-threshold', default=5, type=int)
    parser.add_argument('--start', default=0, type=float)
    parser.add_argument('--end', default=None, type=float)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)
    parser.add_argument('--classifiers', nargs='+', type=str)
    args = parser.parse_args()

    classifiers = []
    for clf in args.classifiers:
        with open(clf, 'rb') as pfile:
            classifiers.append(pickle.load(pfile))

    v_detection = VehicleDetection(classifiers, heatmap_threshold=args.heatmap_threshold, debug_mode=args.debug)
    start = float(args.start)
    end = float(args.end) if args.end is not None else None
    in_video = VideoFileClip(args.input).subclip(t_start=start, t_end=end)
    out_video = in_video.fl_image(v_detection.process_frame)
    out_video.write_videofile(args.output, audio=False)
