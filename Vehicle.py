class Vehicle:

    def __init__(self,
                 box):
        self._x = (box[0][0] + box[1][0]) // 2
        self._y = (box[0][1] + box[1][1]) // 2
        self._y_span = [abs(box[0][1] - box[1][1])]
        self._x_span = [abs(box[0][0] - box[1][0])]
        self._frame_cnt = 1

    def inside_box(self, box):
        return (self._x > box[0][0]) and (self._x < box[1][0]) and \
               (self._y > box[0][1]) and (self._y < box[1][1])

    def update(self, box):
        self._x = (self._x + (box[0][0] + box[1][0]) // 2) // 2
        self._y = (self._y + (box[0][1] + box[1][1]) // 2) // 2
        self._x_span.append(abs(box[0][0] - box[1][0]))
        self._y_span.append(abs(box[0][1] - box[1][1]))
        self._frame_cnt += 1
        return self

    def bounding_box(self):
        frame_avg = 5
        half_x_span = sum(self._x_span[-frame_avg:]) // (2 * frame_avg)
        half_y_span = sum(self._y_span[-frame_avg:]) // (2 * frame_avg)
        left = self._x - half_x_span
        right = self._x + half_x_span
        bottom = self._y - half_y_span
        top = self._y + half_y_span
        return (left, bottom), (right, top)

    def centroid(self):
        return self._x, self._y

    def frame_cnt(self):
        return self._frame_cnt