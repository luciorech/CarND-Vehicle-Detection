import numpy as np
import cv2
from scipy.ndimage.measurements import label
from Vehicle import Vehicle


class VehicleDetection:

    def __init__(self,
                 classifiers,
                 heatmap_threshold,
                 debug_mode=False):
        self._classifiers = classifiers
        self._heatmap_threshold = heatmap_threshold
        self._debug_mode = debug_mode
        self._heatmap = None
        self._vehicles = []

    def process_frame(self, frame):
        # RGB TO BGR
        bgr_frame = np.copy(frame)[...,::-1]

        heatmap = np.zeros_like(bgr_frame[:, :, 0]).astype(np.float)
        hot_windows = []
        for clf in self._classifiers:
            # hot_windows += clf.search_vehicles(bgr_frame, heatmap, self._windows, scale=(64, 64))
            hot_windows += clf.fast_search_vehicles(bgr_frame,
                                                    heatmap,
                                                    x_start_stop=[0, 1279],
                                                    y_start_stop=[400, 719],
                                                    scale=1.25,
                                                    cells_per_step=2)
        if self._heatmap is not None:
            heatmap += self._heatmap
        # 0.9 = cool down factor
        self._heatmap = heatmap * 0.85
        # for hm in self._heatmaps[-3:-1]:
        #     heatmap += hm

        thr_heatmap = np.array(heatmap > self._heatmap_threshold)
        labels = label(thr_heatmap)
        # self._vehicles = self._update_vehicles_list(labels)
        annotated_img = self._draw_labeled_boxes(frame, labels, color=(0, 0, 255), thickness=6)
        # annotated_img = self._draw_vehicles(frame, frame_cnt_threshold=3, color=(0, 0, 255), thickness=6)

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

    def _update_vehicles_list(self,
                              labels):
        new_vehicles = []
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            matches = []
            for v in self._vehicles:
                if v.inside_box(bbox):
                    matches.append(v)
            print("bbox = ", bbox, " matches = ", len(matches), " vehicles = ", len(self._vehicles))
            if len(matches) == 0:
                new_vehicles.append(Vehicle(bbox))
            elif len(matches) == 1:
                new_vehicles.append(v.update(bbox))
            else:
                # todo: recalculate centroids and spans for each match
                new_vehicles.append(matches[0].update(bbox))
                # new_vehicles.append(Vehicle(bbox))
        return new_vehicles

    def _draw_vehicles(self,
                       img,
                       frame_cnt_threshold=2,
                       color=(0, 0, 255),
                       thickness=6):
        draw_img = np.copy(img)
        for v in self._vehicles:
            if v.frame_cnt() >= frame_cnt_threshold:
                ll, ur = v.bounding_box()
                cv2.rectangle(draw_img, ll, ur, color, thickness=thickness)
                x, y = v.centroid()
                cv2.circle(draw_img, (x, y), 5, (0, 255, 0), thickness=3)
            else:
                x, y = v.centroid()
                cv2.circle(draw_img, (x, y), 5, (255, 0, 0), thickness=3)
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
