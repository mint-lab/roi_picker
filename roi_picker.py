import argparse
import json
import numpy as np
import cv2 as cv
import opencx as cx
from random import randrange


POLY_X_AXIS = 30
LINE_X_AXIS = 380
PTS_X_AXIS = 690
Y_AXIS = 20

POLYGON = 'Polygon'
LINE = 'Line'
POINT = 'Point'

POLY_IND = 1
POINT_IND = 2
LINE_IND = 3



class ROIPicker(object):
    def __init__(self, img_file, roi_file='', config_file='roi_picker.json'):
        '''The constructor'''
        self.img_file = img_file
        self.roi_file = roi_file
        self.roi_data = {}
        self.roi_select = ''
        self.is_exit = False
        self.is_drag = False
        self.is_overlap = False
        self.mouse_pt = [0, 0]
        self.config = ROIPicker.load_default_config()

        # Load `config_file` if exist

        # Preprocess variables
        if not self.roi_file:
            ext_idx = self.img_file.rfind('.')
            if ext_idx >= 0:
                self.roi_file = self.img_file[:ext_idx] + '.json'
            else:
                self.roi_file = self.img_file + '.json'


    def load_default_config():
        config = {}

        config['image_scale']           = 1.
        config['image_scale_step']      = 0.05

        config['roi_point_color']       = (0, 255, 0)
        config['roi_point_radius']      = 5
        config['roi_line_color']        = (255, 255, 255)
        config['roi_polygon_alpha']     = 0.4
        config['roi_thickness']         = 1
        config['roi_overlap_radius']    = 5
        config['roi_dist_threshold']    = 0.5

        config['zoom_level']            = 10
        config['zoom_origin_color']     = (0, 0, 255)
        config['zoom_box_radius']       = 10
        config['zoom_box_margin']       = 10

        config['status_font']           = cv.FONT_HERSHEY_DUPLEX
        config['status_font_scale']     = 1
        config['status_font_color']     = (0, 0, 255)
        config['status_thickness']      = 1

        config['key_exit']              = 27
        config['key_roi_select']        = ord('\t')
        config['key_add_point']         = ord('P') - ord('A') + 1
        config['key_add_line']          = ord('L') - ord('A') + 1
        config['key_add_polygon']       = ord('G') - ord('A') + 1
        config['key_renew']             = ord('R') - ord('A') + 1
        config['key_delete_roi']        = ord('D') - ord('A') + 1
        config['key_import']            = ord('I') - ord('A') + 1
        config['key_export']            = ord('E') - ord('A') + 1
        config['key_scale_up']          = ord('+')
        config['key_scale_down']        = ord('-')
        config['key_scale_up2']         = ord('=')
        config['key_scale_down2']       = ord('_')
        return config


    def run_gui(self, wnd_name='ROI Picker', wait_msec=1):
        # Create a window and set a mouse callback to handle mouse events
        cv.namedWindow(wnd_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(wnd_name, self.process_mouse_events)

        # Load `self.img_file`
        img = cv.imread(self.img_file)

        # Load `self.roi_file` if exist

        while not self.is_exit:
            canvas = self.resize_img(img)
            canvas = self.draw_ROIs(canvas)
            canvas = self.viewer_zoom(canvas, self.resize_img(img))
            cv.imshow(wnd_name, canvas)
            key = cv.waitKey(wait_msec)
            self.process_key_events(key)

        return canvas


    def process_mouse_events(self, event, x, y, buttons, user_param):

        if event == cv.EVENT_MOUSEMOVE:
            self.mouse_pt[0] = x
            self.mouse_pt[1] = y

        # When the left mouse button is pressed
        if event == cv.EVENT_LBUTTONDOWN:
            self.is_drag = True
            self.is_overlap, self.ind_point = self.check_overlap(x, y)

        # When the left mouse button is released (It means when a point is dragged)
        elif event == cv.EVENT_LBUTTONUP:
            if self.is_overlap == 1 and self.is_drag == 1:
                self.roi_data[self.roi_select]['pts'][self.ind_point] = (x, y)
            # when the left mouse button is released -> set the flag dragging to false.
            self.is_drag = False

        # When the left mouse button is double clicked
        elif event == cv.EVENT_LBUTTONDBLCLK:
            overlap, ind_point = self.check_overlap(x, y)

            # If  the newly added point already exists, delete it (used for function delete point)
            if overlap:
                self.roi_data[self.roi_select]['pts'].pop(ind_point)
            # Left click means adding a point at current position to the list of points
            else:
                on_a_line, ind_point = self.check_on_a_line(x, y)

                if on_a_line:
                    self.roi_data[self.roi_select]['pts'].insert(ind_point + 1, (x, y))
                elif len(self.roi_data) > 0:
                    self.roi_data[self.roi_select]['pts'].append((x, y))
                else:
                    self.roi_select = 'Polygon_1'
                    self.roi_data[self.roi_select] = {}
                    self.roi_data[self.roi_select]['pts'] = [(x, y)]
                    self.roi_data[self.roi_select]['color'] = randcolor()


    def process_key_events(self, key):
        if key == -1: # No key input
            return

        elif key == self.config['key_exit']:
            self.is_exit = True

        elif key == self.config['key_add_point']:
            roi_type =  POINT
            self.roi_select = roi_type
            if self.roi_select not in self.roi_data.keys():
                self.roi_data[self.roi_select] = {'color': self.randcolor(), 'pts': []}

        elif key == self.config['key_add_line']:
            roi_type = LINE
            self.roi_select = self.make_new_key(roi_type)
            self.roi_data[self.roi_select] = {'color': self.randcolor(), 'pts': []}

        elif key == self.config['key_add_polygon']:
            roi_type = POLYGON
            self.roi_select = self.make_new_key(roi_type)
            self.roi_data[self.roi_select] = {'color': randcolor(), 'pts': []}

        elif key == self.config['key_renew']:
            roi_type = POLYGON
            self.roi_select = roi_type
            self.roi_data = {}

        elif key == self.config['key_delete_roi']:
            self.roi_data[self.roi_select]['pts'] = []

        elif key == self.config['key_import']:
            roi_type = POLYGON
            self.roi_data = self.load_roi()
            self.roi_select = self.make_new_key(roi_type)
            self.roi_data[self.roi_select] = {'color': self.randcolor(), 'pts': []}

        elif key == self.config['key_export']:
            self.save_points()

        elif key == self.config['key_scale_up'] or key == self.config['key_scale_up2']:
            self.config['image_scale'] += self.config['image_scale_step']
            self.resize_roi(self.config['image_scale'] / (self.config['image_scale'] - self.config['image_scale_step']))

        elif key == self.config['key_scale_down'] or key == self.config['key_scale_down2']:
            self.config['image_scale'] -= self.config['image_scale_step']
            self.resize_roi(self.config['image_scale'] / (self.config['image_scale'] + self.config['image_scale_step']))


    def draw_ROIs(self, canvas):
        if (len(self.roi_data) > 0):
            for roi_key in self.roi_data:

                #Draw all points
                for i, axis in enumerate(self.roi_data[roi_key]['pts']):
                        cv.circle(canvas, axis, 2 * self.config['roi_overlap_radius'], self.config['roi_point_color'], 1)
                        cx.putText(canvas, str(i+1), (axis[0] - self.config['roi_overlap_radius'], axis[1] - self.config['roi_overlap_radius']), self.config['status_font'], 0.5, self.config['status_font_color'], self.config['roi_thickness'])

                # Fill polygons and put text
                if roi_key.startswith(POLYGON):
                    poly_ind = int(roi_key.split('_')[1])
                    # Draw all the current polygon segments
                    cv.polylines(canvas, np.array([self.roi_data[roi_key]['pts']]), False, self.config['roi_line_color'], self.config['roi_thickness'])
                    # And points and put text
                    cx.putText(canvas, "ROI (Polygon {num_poly}: {num_point} points)"
                            .format(num_poly = poly_ind, num_point = len(self.roi_data[roi_key]['pts'])),
                            (POLY_X_AXIS, poly_ind * Y_AXIS), self.config['status_font_scale'], 1.5, self.config['status_font_color'], self.config['status_thickness'])
                    # mask to fill polygon
                    if len(self.roi_data[roi_key]['pts']) > 0:
                        mask = canvas.copy()
                        cv.fillPoly(mask, np.array([self.roi_data[roi_key]['pts']]), self.roi_data[roi_key]['color'])
                        cv.addWeighted(mask, 1 - self.config['roi_polygon_alpha'], canvas, self.config['roi_polygon_alpha'], 0, canvas)

                # Draw all lines
                if roi_key.startswith(LINE):
                    poly_ind = int(roi_key.split('_')[1])
                    # Draw all the current polygon segments
                    cv.polylines(canvas, np.array([self.roi_data[roi_key]['pts']]), False, self.config['roi_line_color'], self.config['roi_thickness'])
                    # And points and put text
                    cx.putText(canvas, "ROI (Line {num_line}: {num_point} points)"
                            .format(num_line = poly_ind, num_point = len(self.roi_data[roi_key]['pts'])),
                            (LINE_X_AXIS, poly_ind * Y_AXIS), self.config['status_font_scale'], 1.5, self.config['status_font_color'], self.config['status_thickness'])

                # Put text for all points
                if roi_key.startswith(POINT):
                    cx.putText(canvas, "ROI (Separated Points: {num_point} points)"
                                .format(num_point = len(self.roi_data[roi_key])),
                                (PTS_X_AXIS, Y_AXIS), self.config['status_font_scale'], 1.5, self.config['status_font_color'], self.config['status_thickness'])

        return canvas


    def resize_img(self, img):
        # Calculate the new image dimensions
        new_height = int(img.shape[0] * self.config['image_scale'])
        new_width = int(img.shape[1] * self.config['image_scale'])
        img_resize = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

        return img_resize


    def new_coordinates_after_resize(self, scale, original_coordinate):
        x = int(original_coordinate[0] * scale)
        y = int(original_coordinate[1] * scale)
        return (x, y)


    def resize_roi(self, scale):
        # Calculate the new coordinates
        for key, points in self.roi_data.items():
            for ind in range(len(points['pts'])):
                self.roi_data[key]['pts'][ind] = self.new_coordinates_after_resize(scale, points['pts'][ind])


    def viewer_zoom(self, img, original_img):
        img_height, img_width, *_ = img.shape

        # Update the window
        if self.mouse_pt[0] >= self.config['zoom_box_radius'] and self.mouse_pt[0] < (img_width  - self.config['zoom_box_radius']) and \
        self.mouse_pt[1] >= self.config['zoom_box_radius'] and self.mouse_pt[1] < (img_height - self.config['zoom_box_radius']):
            # Crop the target region
            img_crop = original_img[self.mouse_pt[1] - self.config['zoom_box_radius']:self.mouse_pt[1] + self.config['zoom_box_radius'], \
                        self.mouse_pt[0] - self.config['zoom_box_radius']:self.mouse_pt[0] + self.config['zoom_box_radius'], :]

            # Get the zoomed (resized) image
            zoom_box = cv.resize(img_crop, None, None, self.config['zoom_level'], self.config['zoom_level'])
            height, width, *_ = zoom_box.shape
            x_center, y_center = int(width / 2), int(height / 2)
            # x, y = self.new_coordinates_after_resize_img((width_ori, height_ori), (width_new, height_new), (self.mouse[0], self.mouse[1]))
            cv.line(zoom_box, (x_center, y_center + 10), (x_center, y_center - 10), self.config['zoom_origin_color'], 2)
            cv.line(zoom_box, (x_center + 10, y_center), (x_center - 10, y_center), self.config['zoom_origin_color'], 2)

            # Paste the zoomed image on 'img_copy'
            # s = ZOOM_BOX_MARGIN
            # e = ZOOM_BOX_MARGIN + len(zoom_box)
            s = self.config['zoom_box_margin']
            e =  self.config['zoom_box_margin'] + len(zoom_box)
            s1 = img_width - self.config['zoom_box_margin'] - len(zoom_box)
            e1 = img_width - self.config['zoom_box_margin']

            # img[s:e,s:e,:] = zoom_box
            img[s:e,s1:e1,:] = zoom_box

        return img


    def make_new_key(self, key_count):
        # Count times key_count occurs in polygons_key
        keys = [key.split('_')[0] for key in self.roi_data]
        times = keys.count(key_count)

        return key_count + '_' + str(times+1)


    # Check if a new point was created in the map or not
    def check_overlap(self, x, y):
        overlap = 0
        current_point = 0

        if len(self.roi_data) > 0:
            for key, values in self.roi_data.items():
                for point in values['pts']:
                    # Check if a new point was created in the map or not
                    if pow(x - point[0], 2) + pow(y - point[1], 2) < pow(self.config['roi_overlap_radius'], 2):
                        overlap =  1
                        self.roi_select = key
                        current_point = values['pts'].index(point)
                        break

        return overlap, current_point


    def check_on_a_line(self, x, y):
        on_line = 0
        current_point = 0

        if self.roi_data:
            for key, points in self.roi_data.items():
                if key != POINT:
                    distances = [distance_point2line((x, y), points['pts'][i], points['pts'][(i + 1) % len(points['pts'])]) for i in range(len(points['pts']))]
                    if any(distance < self.config['roi_dist_threshold'] for distance in distances):
                        on_line = 1
                        current_point = distances.index(min(distances))
                        self.roi_select = key

        return on_line, current_point



    # Save all drew points as  josn format
    def save_points(self):
        self.resize_roi(1 / self.config['image_scale'])
        self.config['image_scale'] = 1

        json_object = json.dumps(self.roi_data, indent=4)
        # Writing to sample.json
        with open(self.roi_file, "w") as outfile:
            outfile.write(json_object)


    def load_roi(self):
        data = {}
        file_name = self.img_file.split('/')[-1][:-4] + '.json'
        path_file = os.path.join(self.roi_file, file_name)

        # a dictionary
        if os.path.exists(path_file):
            f = open(path_file)
            data = json.load(f)

        return data


def distance_point2point(pt1, pt2):
    return np.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))


def distance_point2line(pt, line_pt1, line_pt2):
    return distance_point2point(line_pt1, pt) + distance_point2point(line_pt2, pt) - distance_point2point(line_pt1, line_pt2)


def randcolor():
    return (randrange(255), randrange(255), randrange(255))


if __name__ == "__main__":
    # # Add arguments to the parser
    # parser = argparse.ArgumentParser(prog='ROI Picker',
    #                                  description='A simple OpenCV tool to visualize and edit ROIs on images')
    # parser.add_argument('image_file',
    #                     type=str, help='Specify an image file as background')
    # parser.add_argument('roi_file', '-r', default='',
    #                     type=str, help='Specify a ROI file which contains ROI data')
    # parser.add_argument('config_file', '-c', default='roi_picker.json',
    #                     type=str, help='Specify a configuration file')

    # # Parse the command-line arguments
    # args = parser.parse_args()

    # # Instantiate ROI Picker and run it
    # roi_picker = ROIPicker(args.image_file, args.roi_file, args.config_file)
    roi_picker = ROIPicker('demo/miraehall_220722_gray.jpg')
    roi_picker.run_gui()