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
        self._debug_file = None
        self._frame_cnt = 0

    def process_frame(self, frame):
        if self._debug_file is None:
            self._debug_file = open('./debug/log', 'w')
        self._frame_cnt += 1

        # RGB TO BGR
        bgr_frame = np.copy(frame)[...,::-1]

        heatmap = np.zeros_like(bgr_frame[:, :, 0]).astype(np.float)
        hot_windows = []
        for clf in self._classifiers:
            hot_windows += clf.fast_search_vehicles(bgr_frame,
                                                    heatmap,
                                                    x_start_stop=[0, 1279],
                                                    y_start_stop=[400, 520],
                                                    scale=1.3,
                                                    cells_per_step=2)
            hot_windows += clf.fast_search_vehicles(bgr_frame,
                                                    heatmap,
                                                    x_start_stop=[0, 1279],
                                                    y_start_stop=[400, 719],
                                                    scale=2,
                                                    cells_per_step=3)

        if self._heatmap is not None:
            heatmap += self._heatmap
        self._heatmap = heatmap * 0.90

        thr_heatmap = np.array(heatmap > self._heatmap_threshold)
        labels = label(thr_heatmap)
        self._vehicles = self._update_vehicles_list(labels)
        annotated_img = self._draw_vehicles(frame, frame_cnt_threshold=5, color=(0, 0, 255), thickness=6)

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
            cv2.imwrite("./debug/%s.jpg" % self._frame_cnt, result[...,::-1])
        else:
            result = annotated_img

        return result

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
        if self._debug_mode:
            self._debug_file.write("**** Frame %s \n" % self._frame_cnt)
        for i in range(len(self._vehicles)):
            v = self._vehicles[i]
            if self._debug_mode:
                self._debug_file.write("{0} Vehicle: x = {1}, y = {2}, x_span = {3}, y_span = {4}\n".format(i,
                                                                                                            v.centroid()[0],
                                                                                                            v.centroid()[1],
                                                                                                            v.x_span(),
                                                                                                            v.y_span()))
        new_vehicles = []
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            matches = []
            for i in range(len(self._vehicles)):
                v = self._vehicles[i]
                if v.inside_box(bbox):
                    matches.append(v)
                    if self._debug_mode:
                        self._debug_file.write("Vehicle %s is a match.\n" % i)
            self._debug_file.write("bbox = {0}, matches = {1}\n".format(bbox, len(matches)))
            if len(matches) == 0:
                new_vehicles.append(Vehicle(bbox))
            elif len(matches) == 1:
                new_vehicles.append(matches[0].update(bbox))
            else:
                new_vehicles.append(Vehicle(bbox))

        if self._debug_mode:
            for v in new_vehicles:
                self._debug_file.write("New vehicle: x = {0}, y = {1}, x_span = {2}, y_span = {3}\n".format(v.centroid()[0],
                                                                                                            v.centroid()[1],
                                                                                                            v.x_span(),
                                                                                                            v.y_span()))
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
                cv2.circle(draw_img, (x, y), 5, (255, 0, 0), thickness=1)
        return draw_img
