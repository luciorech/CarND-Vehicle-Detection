import numpy as np
import cv2
from scipy.ndimage.measurements import label

class VehicleDetection:

    def __init__(self,
                 classifiers,
                 debug_mode=False):
        self._classifiers = classifiers
        self._debug_mode = debug_mode
        self._windows = []
        self._heatmaps = []

    def create_sliding_windows(self,
                               x_start_stop=[None, None],
                               y_start_stop=[None, None]):
        self._windows = []
        self._windows += self._slide_window(x_start_stop=x_start_stop,
                                            y_start_stop=y_start_stop,
                                            xy_window=(64, 64),
                                            xy_overlap=(0.5, 0.5))
        self._windows += self._slide_window(x_start_stop=x_start_stop,
                                            y_start_stop=y_start_stop,
                                            xy_window=(96, 96),
                                            xy_overlap=(0.5, 0.5))
        self._windows += self._slide_window(x_start_stop=x_start_stop,
                                            y_start_stop=y_start_stop,
                                            xy_window=(128, 128),
                                            xy_overlap=(0.5, 0.5))
        print("Total search windows = {0}".format(len(self._windows)))

    def process_frame(self, frame):
        if len(self._windows) == 0:
            raise Exception('Sliding windows undefined')

        heatmap = np.zeros_like(frame[:, :, 0])
        hot_windows = []
        for clf in self._classifiers:
            hot_windows += clf.search_vehicles(frame, heatmap, self._windows, scale=(64, 64))
        self._heatmaps.append(heatmap)

        for hm in self._heatmaps[-3:-1]:
            heatmap += hm

        thr_heatmap = np.array(heatmap > 4)
        labels = label(thr_heatmap)
        annotated_img = self._draw_labeled_boxes(frame, labels, color=(0, 0, 255), thickness=6)

        if self._debug_mode:
            result = np.zeros_like(frame)
            x_size = frame.shape[1] // 2
            y_size = frame.shape[0] // 2
            boxes = self._draw_boxes(frame, hot_windows, color=(0, 0, 255), thickness=1)
            boxes = cv2.resize(boxes, (x_size, y_size))
            result[y_size:, x_size:, :] = boxes
            scaled_heatmap = cv2.resize(heatmap, (x_size, y_size)) * 32
            result[:y_size, x_size:, :] = cv2.merge((scaled_heatmap, scaled_heatmap, scaled_heatmap))
            scaled_labels = cv2.resize(np.array(labels[0], dtype=heatmap.dtype), (x_size, y_size))
            scaled_labels *= (255 // (labels[1] + 1))
            result[y_size:, :x_size, :] = cv2.merge((scaled_labels, scaled_labels, scaled_labels))
            scaled_ann = cv2.resize(annotated_img, (x_size, y_size))
            result[:y_size, :x_size, :] = scaled_ann
        else:
            result = annotated_img

        return result

    def _slide_window(self,
                      x_start_stop,
                      y_start_stop,
                      xy_window=(64, 64),
                      xy_overlap=(0.5, 0.5)):
        """ Computes sliding window boxes """
        x_span = x_start_stop[1] - x_start_stop[0]
        y_span = y_start_stop[1] - y_start_stop[0]

        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((x_span - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((y_span - ny_buffer) / ny_pix_per_step)

        window_list = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                start_x = xs * nx_pix_per_step + x_start_stop[0]
                end_x = start_x + xy_window[0]
                start_y = ys * ny_pix_per_step + y_start_stop[0]
                end_y = start_y + xy_window[1]
                window_list.append(((start_x, start_y), (end_x, end_y)))
        return window_list

    def _draw_boxes(self,
                    img,
                    bboxes,
                    color=(0, 0, 255),
                    thickness=6):
        draw_img = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thickness)
        return draw_img

    def _draw_labeled_boxes(self,
                            img,
                            labels,
                            color=(0, 0, 255),
                            thickness=6):
        draw_img = np.copy(img)
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thickness)
        return draw_img
