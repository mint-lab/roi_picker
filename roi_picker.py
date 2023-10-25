import argparse
import json
import numpy as np
import cv2 as cv
import opencx as cx
from random import randrange


class ROIPicker:
    '''ROI Picker implementation'''

    def __init__(self, img_file, roi_file='', config_file='roi_picker.json'):
        '''The constructor'''
        self.img_file = img_file
        self.roi_file = roi_file
        self.config_file = config_file
        self.roi_data = []
        self.roi_select_idx = -1
        self.mouse_pt = [0, 0]
        self.resize_canvas = True
        self.redraw_canvas = True
        self.drag_enabled = False
        self.drag_pt_idx = -1
        self.config = self.get_default_config()

        # Load `config_file` if exist
        try:
            config = read_json_file(config_file)
            self.config.update(config)
        except FileNotFoundError:
            print('warning: cannot open the configuration file, ' + config_file)
            print('         ROI Picker will use the default configuration.')

        # Preprocess variables
        if not self.roi_file:
            ext_idx = self.img_file.rfind('.')
            if ext_idx >= 0:
                self.roi_file = self.img_file[:ext_idx] + '.json'
            else:
                self.roi_file = self.img_file + '.json'

        self.config_selected = self.config.copy()
        self.config_selected['roi_point_thickness']   = self.config['roi_select_thickness']
        self.config_selected['roi_line_thickness']    = self.config['roi_select_thickness']
        self.config_selected['roi_polygon_thickness'] = self.config['roi_select_thickness']


    @staticmethod
    def get_default_config():
        '''Generate the default configuration'''
        config = {}

        config['image_scale']           = 1.
        config['image_scale_step']      = 0.05

        config['point_dist_threshold']  = 5
        config['line_dist_threshold']   = 2

        config['roi_id_start']          = 1
        config['roi_id_offset']         = (-5, -7)
        config['roi_id_font']           = cv.FONT_HERSHEY_DUPLEX
        config['roi_id_font_scale']     = 0.5
        config['roi_id_font_thickness'] = 1
        config['roi_point_radius']      = 10
        config['roi_point_thickness']   = 1
        config['roi_line_thickness']    = 2
        config['roi_polygon_alpha']     = 0.7
        config['roi_polygon_thickness'] = 1
        config['roi_select_thickness']  = 2

        config['zoom_visible']          = True
        config['zoom_level']            = 10
        config['zoom_box_radius']       = 10
        config['zoom_box_margin']       = 10
        config['zoom_axes_radius']      = 20
        config['zoom_axes_color']       = (0, 0, 255)
        config['zoom_axes_thickness']   = 1

        config['status_visible']        = True
        config['status_offset']         = (10, 10)
        config['status_font']           = cv.FONT_HERSHEY_DUPLEX
        config['status_font_scale']     = 0.6
        config['status_font_thickness'] = 1

        config['key_exit']              = 27
        config['key_next_roi']          = ord('\t')
        config['key_add_point']         = ord('P') - ord('A') + 1
        config['key_add_line']          = ord('L') - ord('A') + 1
        config['key_add_polygon']       = ord('G') - ord('A') + 1
        config['key_renew']             = ord('R') - ord('A') + 1
        config['key_delete_roi']        = ord('D') - ord('A') + 1
        config['key_import_roi']        = ord('I') - ord('A') + 1
        config['key_export_roi']        = ord('E') - ord('A') + 1
        config['key_export_config']     = ord('F') - ord('A') + 1
        config['key_show_zoom']         = ord('Z') - ord('A') + 1
        config['key_show_status']       = ord('T') - ord('A') + 1
        config['key_scale_up']          = ord('+')
        config['key_scale_up2']         = ord('=')
        config['key_scale_down']        = ord('-')
        config['key_scale_down2']       = ord('_')
        return config


    def run_gui(self, wnd_name='ROI Picker', wait_msec=1):
        '''Run 'ROI Picker' with OpenCV GUI'''
        # Create a window and register a callback function to handle mouse events
        cv.namedWindow(wnd_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(wnd_name, self.process_mouse_events)

        # Load `self.img_file`
        img = cv.imread(self.img_file)
        if img is None:
            print('error: cannot open the given image file, ' + self.img_file)
            return False

        # Load `self.roi_file` if exist
        try:
            self.roi_data = self.import_roi_data(self.roi_file)
        except FileNotFoundError:
            print('warning: cannot open the corresponding ROI file, ' + self.roi_file)
        self.roi_select_idx = len(self.roi_data) - 1

        # Run 'ROI Picker'
        while True:
            # Draw the current ROIs
            # Note) The following implementation tries to avoid unnecessary resize and ROI drawing for better response.
            if self.resize_canvas:
                img_resize = cv.resize(img, None, None, self.config['image_scale'], self.config['image_scale'])
                self.resize_canvas = False
                self.redraw_canvas = True
            if self.redraw_canvas:
                canvas = np.copy(img_resize)
                for roi in self.roi_data:
                    self.draw_roi(canvas, roi, self.config)
                if self.roi_select_idx >= 0:
                    self.draw_roi(canvas, self.roi_data[self.roi_select_idx], self.config_selected)
                self.redraw_canvas = False
            canvas_copy = np.copy(canvas)
            if self.config['zoom_visible']:
                self.draw_zoom(canvas_copy, img_resize)
            if self.config['status_visible']:
                self.print_status(canvas_copy)

            # Show `canvas` and process user inputs
            cv.imshow(wnd_name, canvas_copy)
            key = cv.waitKey(wait_msec)
            need_exit = self.process_key_inputs(key)
            if need_exit:
                break

        return True


    def process_mouse_events(self, event, x, y, buttons, user_param):
        '''Process mouse events'''
        if event == cv.EVENT_MOUSEMOVE:
            self.mouse_pt[0] = x
            self.mouse_pt[1] = y

        if self.roi_select_idx >= 0:
            if event == cv.EVENT_LBUTTONDOWN:
                # Start to drag a selected point
                self.drag_enabled = buttons & cv.EVENT_FLAG_CTRLKEY
                self.drag_pt_idx = check_on_a_point((x, y), self.roi_data[self.roi_select_idx]['pts'], self.config['point_dist_threshold'])

            elif event == cv.EVENT_LBUTTONUP:
                # Finish dragging the point
                if self.drag_enabled and self.drag_pt_idx >= 0:
                    self.roi_data[self.roi_select_idx]['pts'][self.drag_pt_idx] = (x, y)
                self.drag_enabled = False
                self.drag_pt_idx = -1
                self.redraw_canvas = True

            elif event == cv.EVENT_LBUTTONDBLCLK:
                on_a_point = check_on_a_point((x, y), self.roi_data[self.roi_select_idx]['pts'], self.config['point_dist_threshold'])
                if on_a_point >= 0:
                    # Remove a selected point
                    self.roi_data[self.roi_select_idx]['pts'].pop(on_a_point)
                else:
                    if self.roi_data[self.roi_select_idx]['type'].lower().startswith('points'):
                        # Add a new point if the selected ROI is 'points'
                        self.roi_data[self.roi_select_idx]['pts'].append((x, y))
                    else:
                        if self.roi_data[self.roi_select_idx]['type'].lower().startswith('line') and len(self.roi_data[self.roi_select_idx]['pts']) > 0:
                            on_a_line = check_on_a_line((x, y), self.roi_data[self.roi_select_idx]['pts'] + [self.roi_data[self.roi_select_idx]['pts'][0]], self.config['line_dist_threshold'])
                        else:
                            on_a_line = check_on_a_line((x, y), self.roi_data[self.roi_select_idx]['pts'], self.config['line_dist_threshold'])
                        if on_a_line >= 0:
                            # Insert a new point between a selected line segment
                            self.roi_data[self.roi_select_idx]['pts'].insert(on_a_line, (x, y))
                        else:
                            # Add a new point at the end
                            self.roi_data[self.roi_select_idx]['pts'].append((x, y))
                self.redraw_canvas = True


    def process_key_inputs(self, key):
        '''Process key inputs and return `True` if the program need to exit'''
        if key == -1: # No key input
            pass

        elif key == self.config['key_exit']:
            return True

        elif key == self.config['key_next_roi']:
            if len(self.roi_data) == 0:
                self.roi_select_idx = -1
            else:
                self.roi_select_idx = (self.roi_select_idx + 1) % len(self.roi_data)
            self.redraw_canvas = True

        elif key == self.config['key_add_point']:
            self.roi_data.append({'type' : 'points', 'pts'  : [], 'color': randcolor()})
            self.roi_select_idx = len(self.roi_data) - 1
            self.redraw_canvas = True

        elif key == self.config['key_add_line']:
            self.roi_data.append({'type' : 'line', 'pts'  : [], 'color': randcolor()})
            self.roi_select_idx = len(self.roi_data) - 1
            self.redraw_canvas = True

        elif key == self.config['key_add_polygon']:
            self.roi_data.append({'type' : 'polygon', 'pts'  : [], 'color': randcolor()})
            self.roi_select_idx = len(self.roi_data) - 1
            self.redraw_canvas = True

        elif key == self.config['key_renew']:
            self.roi_data.clear()
            self.redraw_canvas = True

        elif key == self.config['key_delete_roi']:
            if self.roi_select_idx >= 0:
                self.roi_data.pop(self.roi_select_idx)
            self.roi_select_idx = len(self.roi_data) - 1
            self.redraw_canvas = True

        elif key == self.config['key_import_roi']:
            try:
                self.roi_data = self.import_roi_data(self.roi_file)
            except FileNotFoundError:
                print('warning: cannot open the corresponding ROI file, ' + self.roi_file)
            self.roi_select_idx = len(self.roi_data) - 1

        elif key == self.config['key_export_roi']:
            self.export_roi_data(self.roi_file, self.roi_data)

        elif key == self.config['key_export_config']:
            write_json_file(self.config_file, self.config)

        elif key == self.config['key_show_zoom']:
            self.config['zoom_visible'] = not self.config['zoom_visible']

        elif key == self.config['key_show_status']:
            self.config['status_visible'] = not self.config['status_visible']

        elif key == self.config['key_scale_up'] or key == self.config['key_scale_up2']:
            original_scale = self.config['image_scale']
            self.config['image_scale'] += self.config['image_scale_step']
            self.resize_roi_data(self.roi_data, self.config['image_scale'] / original_scale)
            self.resize_canvas = True
            self.redraw_canvas = True

        elif key == self.config['key_scale_down'] or key == self.config['key_scale_down2']:
            original_scale = self.config['image_scale']
            self.config['image_scale'] -= self.config['image_scale_step']
            self.resize_roi_data(self.roi_data, self.config['image_scale'] / original_scale)
            self.resize_canvas = True
            self.redraw_canvas = True

        return False


    def draw_roi(self, canvas, roi, config):
        '''Draw the given `roi` with the given `config`'''
        if roi['type'].lower().startswith('line'):
            # Draw a line segment
            if config['roi_line_thickness'] > 0:
                cv.polylines(canvas, np.array([roi['pts']]).astype(np.int32), False, roi['color'], config['roi_line_thickness'])

        elif roi['type'].lower().startswith('polygon'):
            # Draw a polygon
            if config['roi_polygon_alpha'] < 1 and len(roi['pts']) > 2:
                polygon = canvas.copy()
                cv.fillPoly(polygon, np.array([roi['pts']]).astype(np.int32), roi['color'])
                cv.addWeighted(polygon, 1 - config['roi_polygon_alpha'], canvas, config['roi_polygon_alpha'], 0, canvas)
            if config['roi_polygon_thickness'] > 0:
                cv.polylines(canvas, np.array([roi['pts']]).astype(np.int32), True, roi['color'], config['roi_polygon_thickness'])

        for idx, pt in enumerate(roi['pts']):
            # Draw a point with its ID
            center = np.array(pt).astype(np.int32)
            if config['roi_point_radius'] > 0:
                cv.circle(canvas, center, config['roi_point_radius'], roi['color'], config['roi_point_thickness'])
            if config['roi_id_font_scale'] > 0:
                id = idx + config['roi_id_start']
                id_pos = center + config['roi_id_offset']
                if id >= 10: # If `id` is two-digit
                    id_pos[0] += config['roi_id_offset'][0]
                color_inv = [255 - v for v in roi['color']]
                cx.putText(canvas, str(id), id_pos, config['roi_id_font'], config['roi_id_font_scale'], roi['color'], config['roi_id_font_thickness'], color_inv)
            else:
                cv.line(canvas, center - (config['roi_point_radius'], 0), center + (config['roi_point_radius'], 0), roi['color'], config['roi_id_font_thickness'])
                cv.line(canvas, center - (0, config['roi_point_radius']), center + (0, config['roi_point_radius']), roi['color'], config['roi_id_font_thickness'])


    def draw_zoom(self, canvas, image):
        '''Draw the zoomed image pointing by the mouse cursor, `self.mouse_pt`'''
        h, w, *_ = canvas.shape
        if self.mouse_pt[0] >= self.config['zoom_box_radius'] and self.mouse_pt[0] < (w - self.config['zoom_box_radius']) and \
           self.mouse_pt[1] >= self.config['zoom_box_radius'] and self.mouse_pt[1] < (h - self.config['zoom_box_radius']):
            # Crop the target region
            img_crop = image[self.mouse_pt[1] - self.config['zoom_box_radius']:self.mouse_pt[1] + self.config['zoom_box_radius'], \
                             self.mouse_pt[0] - self.config['zoom_box_radius']:self.mouse_pt[0] + self.config['zoom_box_radius'], :]

            # Make the zoomed (resized) image
            img_zoom = cv.resize(img_crop, None, None, self.config['zoom_level'], self.config['zoom_level'])
            h_zoom, w_zoom, *_ = img_zoom.shape
            if self.config['zoom_axes_thickness'] > 0:
                center = np.array((w_zoom / 2, h_zoom / 2)).astype(np.int32)
                cv.line(img_zoom, center - (self.config['zoom_axes_radius'], 0), center + (self.config['zoom_axes_radius'], 0), self.config['zoom_axes_color'], self.config['zoom_axes_thickness'])
                cv.line(img_zoom, center - (0, self.config['zoom_axes_radius']), center + (0, self.config['zoom_axes_radius']), self.config['zoom_axes_color'], self.config['zoom_axes_thickness'])

            # Paste the zoomed image on 'img_copy'
            ys = self.config['zoom_box_margin']
            ye = self.config['zoom_box_margin'] + h_zoom
            xs = w - self.config['zoom_box_margin'] - w_zoom
            xe = w - self.config['zoom_box_margin']
            canvas[ys:ye,xs:xe,:] = img_zoom


    def print_status(self, canvas):
        '''Print the status of the selected ROI'''
        if self.roi_select_idx >= 0:
            roi = self.roi_data[self.roi_select_idx]
            status = f'{roi["type"]}: {len(roi["pts"])} points'
            status_color = roi['color']
        else:
            status = 'Empty ROI'
            status_color = (0, 255, 0)
        pt = (self.mouse_pt[0] / self.config['image_scale'], self.mouse_pt[1] / self.config['image_scale'])
        status += f' | mouse: ({pt[0]:.1f}, {pt[1]:.1f})'
        status += f' | zoom: {self.config["image_scale"]:.1f}'
        cx.putText(canvas, status, self.config['status_offset'], self.config['status_font'], self.config['status_font_scale'], status_color, self.config['status_font_thickness'])


    @staticmethod
    def resize_roi_data(roi_data, scale):
        '''Resize points in `roi_data` with `scale` factor'''
        for roi in roi_data:
            for idx, pt in enumerate(roi['pts']):
                roi['pts'][idx] = (scale * pt[0], scale * pt[1])


    def import_roi_data(self, filename):
        '''Import ROI data from the file `filename`'''
        data = read_json_file(filename)
        self.resize_roi_data(data, self.config['image_scale'])
        return data


    def export_roi_data(self, filename, roi_data):
        '''Export `roi_data` to the file `filename`'''
        data = roi_data.copy()
        self.resize_roi_data(data, 1 / self.config['image_scale'])
        write_json_file(filename, data)


def distance_point2point(pt1, pt2):
    '''Calculate distacne from `pt1` to `pt2`'''
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    return np.sqrt(dx*dx + dy*dy)


def distance_point2line(pt, line_pt1, line_pt2):
    '''Calculate distacne from `pt` to a line segment `(line_pt1, line_pt2)`'''
    return distance_point2point(line_pt1, pt) + distance_point2point(line_pt2, pt) - distance_point2point(line_pt1, line_pt2)


def check_on_a_point(query, pts, threshold):
    '''Check whether `query` is on any point in `pts`, return its index if then'''
    min_idx = -1
    min_dist = 10000
    for idx, pt in enumerate(pts):
        dist = distance_point2point(query, pt)
        if dist < min_dist:
            min_idx = idx
            min_dist = dist
    if min_dist <= threshold:
        return min_idx
    else:
        return -1


def check_on_a_line(query, pts, threshold):
    '''Check whether `query` is on any line made by `pts`, return its first index if then'''
    min_idx = -1
    min_dist = 10000
    for idx in range(len(pts) - 1):
        dist = distance_point2line(query, pts[idx], pts[idx-1])
        if dist < min_dist:
            min_idx = idx
            min_dist = dist
    if min_dist <= threshold:
        return min_idx
    else:
        return -1


def randcolor():
    '''Return a random color'''
    return (randrange(255), randrange(255), randrange(255))


def read_json_file(filename):
    '''Read an object from the given JSON file `filename`'''
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


def write_json_file(filename, data):
    '''Write the given object to the given file `filename`'''
    json_obj = json.dumps(data, indent=4)
    with open(filename, 'w') as f:
        f.write(json_obj)


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
    roi_picker = ROIPicker('demo/miraehall_220722.png')
    success = roi_picker.run_gui()
    # if not success:
    #     parser.print_help()