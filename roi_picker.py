import numpy as np
import cv2 as cv
import json
import math
import argparse
import json
import os
import opencx as cx
from random import randrange


RADIUS = 5
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)
FONT = cv.FONT_HERSHEY_DUPLEX
FONTSCALE = 1 # FONTSCALE
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
THICKNESS = 1 # Line thickness of 1 px
POLY_X_AXIS = 30
LINE_X_AXIS = 380
PTS_X_AXIS = 690
Y_AXIS = 20

KEY_CTRL_M = ord('M') - ord('A') + 1 # CTRL+M: Toggle polygon mode while removing all points
KEY_CTRL_N = ord('N') - ord('A') + 1 # CTRL+N: Remove all points for the current ROI
KEY_CTRL_A = ord('A') - ord('A') + 1 # CTRL+A: Add a new ROI
KEY_CTRL_D = ord('D') - ord('A') + 1 # CTRL+D: Delete the current ROI
KEY_CTRL_X = ord('X') - ord('A') + 1 # CTRL+S: Save(append) ROIs to `roi_file_name.json`
KEY_CTRL_Q = ord('Q') - ord('A') + 1 # CTRL+Q: Add new point
KEY_CTRL_W = ord('W') - ord('A') + 1 # CTRL+W: Add new line
KEY_CTRL_E = ord('E') - ord('A') + 1 # CTRL+Q: Scale up 
KEY_CTRL_R = ord('R') - ord('A') + 1 # CTRL+R: Scale down 
KEY_CTRL_L = ord('L') - ord('A') + 1 # CTRL+L: Load ROI
ESC = 27                             # Escape ROI

POLYGON = 'Polygon'
LINE = 'Line'
POINT = 'Point'
WINDOWNAME = "ROI_PICKER" # Name for our window

POLY_IND = 1
POINT_IND = 2
LINE_IND = 3

ROI_WEIGHT = 0.4
THRESHOLD = 0.5
ZOOM_LEVEL = 10
ZOOM_BOX_RADIUS = 10 
ZOOM_BOX_MARGIN = 10
SCALE_DEFAULT = 1
SCALE_STEP = 0.05

class PolygonDrawer(object):
    def __init__(self, img_load_path, result_folder):
        self.img_load_path = img_load_path
        self.result_folder = result_folder
        self.roi_dict = {}
        self.done = False # Flag signalling we're done
        self.dragging = False # Flag to check drag a point or not
        self.overlap = False # Flag to check whether a new added point is exist or not   
        self.scale = SCALE_DEFAULT
        self.mouse = [0, 0]
        self.current_key = ''
        
               
    def mouse_events(self, event, x, y, buttons, user_param):

        if event == cv.EVENT_MOUSEMOVE:
            self.mouse[0] = x 
            self.mouse[1] = y

        # When the left mouse button is pressed
        if event == cv.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.overlap, self.ind_point = self.check_overlap(x, y)
        
        # When the left mouse button is released (It means when a point is dragged)
        elif event == cv.EVENT_LBUTTONUP:
            if self.overlap == 1 and self.dragging == 1:
                self.roi_dict[self.current_key]['pts'][self.ind_point] = (x, y)
            # when the left mouse button is released -> set the flag dragging to false.
            self.dragging = False
            
        # When the left mouse button is double clicked
        elif event == cv.EVENT_LBUTTONDBLCLK:   
            overlap, ind_point = self.check_overlap(x, y)

            # If  the newly added point already exists, delete it (used for function delete point)
            if overlap:
                self.roi_dict[self.current_key]['pts'].pop(ind_point)
            # Left click means adding a point at current position to the list of points
            else: 
                on_a_line, ind_point = self.check_on_a_line(x, y)

                if on_a_line:
                    self.roi_dict[self.current_key]['pts'].insert(ind_point + 1, (x, y))
                elif len(self.roi_dict) > 0:
                    self.roi_dict[self.current_key]['pts'].append((x, y))
                else:
                    self.current_key = 'Polygon_1'
                    self.roi_dict[self.current_key] = {}
                    self.roi_dict[self.current_key]['pts'] = [(x, y)]
                    self.roi_dict[self.current_key]['color'] = self.random_color()


    def roi_process(self):
        # Create a working window and set a mouse callback to handle events
        cv.namedWindow(WINDOWNAME, cv.WINDOW_AUTOSIZE) 
        cv.waitKey(1)
        cv.setMouseCallback(WINDOWNAME, self.mouse_events)

        img = cv.imread(self.img_load_path)

        while(not self.done):
            canvas= self.resize_img(img)
            canvas = self.draw_polygon(canvas)
            canvas = self.viewer_zoom(canvas, self.resize_img(img)) 
            cv.imshow(WINDOWNAME, canvas)

            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            key =  cv.waitKey(50)

            # CTRL+A: Add a new ROI
            if key == KEY_CTRL_A: 
                roi_type = POLYGON
                self.current_key = self.make_new_key(roi_type)
                self.roi_dict[self.current_key] = {'color': self.random_color(), 'pts': []}

            
            # CTRL+W: Add new line
            if key == KEY_CTRL_W: 
                roi_type = LINE
                self.current_key = self.make_new_key(roi_type)
                self.roi_dict[self.current_key] = {'color': self.random_color(), 'pts': []}


            # CTRL+Q: Add a new point
            if key == KEY_CTRL_Q: 
                roi_type =  POINT
                self.current_key = roi_type
                if self.current_key not in self.roi_dict.keys():
                    self.roi_dict[self.current_key] = {'color': self.random_color(), 'pts': []}


            # CTRL+N: Remove all points
            if key == KEY_CTRL_N: 
                roi_type = POLYGON
                self.current_key = roi_type
                self.roi_dict = {}


            # CTRL+D: Remove all points for the current ROI
            if key == KEY_CTRL_D: 
                self.roi_dict[self.current_key]['pts'] = []

            # CTRL+E: Scale up
            if key == KEY_CTRL_E:
                self.scale += SCALE_STEP
                self.resize_roi(self.scale / (self.scale - SCALE_STEP))


            # CTRL+R: Scale down
            if key == KEY_CTRL_R:
                self.scale -= SCALE_STEP
                self.resize_roi(self.scale / (self.scale + SCALE_STEP))


            # CTRL+X: Save ROIs
            if key == KEY_CTRL_X: 
                self.save_points()


            # CTRL+L: Load ROI
            if key == KEY_CTRL_L:
                roi_type = POLYGON
                self.roi_dict = self.load_roi()
                self.current_key = self.make_new_key(roi_type)
                self.roi_dict[self.current_key] = {'color': self.random_color(), 'pts': []}
        

            # Escape 
            if key == ESC: 
                self.done = True
        
        return canvas
    

    def draw_polygon(self, canvas):

        if (len(self.roi_dict) > 0):
            for roi_key in self.roi_dict:

                #Draw all points
                for i, axis in enumerate(self.roi_dict[roi_key]['pts']): 
                        cv.circle(canvas, axis, 2 * RADIUS, GREEN, 1)
                        cx.putText(canvas, str(i+1), (axis[0] - RADIUS, axis[1] - RADIUS), FONT, 0.5, RED, THICKNESS)
                
                # Fill polygons and put text
                if roi_key.startswith(POLYGON):
                    poly_ind = int(roi_key.split('_')[1])
                    # Draw all the current polygon segments
                    cv.polylines(canvas, np.array([self.roi_dict[roi_key]['pts']]), False, FINAL_LINE_COLOR, THICKNESS)
                    # And points and put text                   
                    cx.putText(canvas, "ROI (Polygon {num_poly}: {num_point} points)"
                            .format(num_poly = poly_ind, num_point = len(self.roi_dict[roi_key]['pts'])), 
                            (POLY_X_AXIS, poly_ind * Y_AXIS), FONTSCALE, 1.5, RED, THICKNESS)
                    # mask to fill polygon
                    if len(self.roi_dict[roi_key]['pts']) > 0:
                        mask = canvas.copy()
                        cv.fillPoly(mask, np.array([self.roi_dict[roi_key]['pts']]), self.roi_dict[roi_key]['color'])
                        cv.addWeighted(mask, 1 - ROI_WEIGHT, canvas, ROI_WEIGHT, 0, canvas)
                
                # Draw all lines
                if roi_key.startswith(LINE):
                    poly_ind = int(roi_key.split('_')[1])
                    # Draw all the current polygon segments
                    cv.polylines(canvas, np.array([self.roi_dict[roi_key]['pts']]), False, FINAL_LINE_COLOR, THICKNESS)
                    # And points and put text
                    cx.putText(canvas, "ROI (Line {num_line}: {num_point} points)"
                            .format(num_line = poly_ind, num_point = len(self.roi_dict[roi_key]['pts'])), 
                            (LINE_X_AXIS, poly_ind * Y_AXIS), FONTSCALE, 1.5, RED, THICKNESS)
                
                # Put text for all points
                if roi_key.startswith(POINT):
                    cx.putText(canvas, "ROI (Separated Points: {num_point} points)"
                                .format(num_point = len(self.roi_dict[roi_key])), 
                                (PTS_X_AXIS, Y_AXIS), FONTSCALE, 1.5, RED, THICKNESS)
            
        return canvas


    def resize_img(self, img):    
        # Calculate the new image dimensions
        new_height = int(img.shape[0] * self.scale)
        new_width = int(img.shape[1] * self.scale)
        img_resize = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
            
        return img_resize
    
    
    def new_coordinates_after_resize(self, scale, original_coordinate):
        x = int(original_coordinate[0] * scale)
        y = int(original_coordinate[1] * scale)
        return (x, y)

  
    def resize_roi(self, scale):    
        # Calculate the new coordinates
        for key, points in self.roi_dict.items():
            for ind in range(len(points['pts'])):
                self.roi_dict[key]['pts'][ind] = self.new_coordinates_after_resize(scale, points['pts'][ind])        


    def viewer_zoom(self, img, original_img):
        img_height, img_width, *_ = img.shape

        # Update the window
        if self.mouse[0] >= ZOOM_BOX_RADIUS and self.mouse[0] < (img_width  - ZOOM_BOX_RADIUS) and \
        self.mouse[1] >= ZOOM_BOX_RADIUS and self.mouse[1] < (img_height - ZOOM_BOX_RADIUS):
            # Crop the target region
            img_crop = original_img[self.mouse[1] - ZOOM_BOX_RADIUS:self.mouse[1] + ZOOM_BOX_RADIUS, \
                        self.mouse[0] - ZOOM_BOX_RADIUS:self.mouse[0] + ZOOM_BOX_RADIUS, :]

            # Get the zoomed (resized) image
            zoom_box = cv.resize(img_crop, None, None, ZOOM_LEVEL, ZOOM_LEVEL)
            height, width, *_ = zoom_box.shape
            x_center, y_center = int(width / 2), int(height / 2)
            # x, y = self.new_coordinates_after_resize_img((width_ori, height_ori), (width_new, height_new), (self.mouse[0], self.mouse[1])) 
            cv.line(zoom_box, (x_center, y_center + 10), (x_center, y_center - 10), RED, 2)
            cv.line(zoom_box, (x_center + 10, y_center), (x_center - 10, y_center), RED, 2)
        
            # Paste the zoomed image on 'img_copy'
            # s = ZOOM_BOX_MARGIN
            # e = ZOOM_BOX_MARGIN + len(zoom_box)
            s = ZOOM_BOX_MARGIN  
            e =  ZOOM_BOX_MARGIN + len(zoom_box)
            s1 = img_width - ZOOM_BOX_MARGIN - len(zoom_box)  
            e1 = img_width - ZOOM_BOX_MARGIN

            # img[s:e,s:e,:] = zoom_box
            img[s:e,s1:e1,:] = zoom_box
        
        return img


    def make_new_key(self, key_count):
        # Count times key_count occurs in polygons_key
        keys = [key.split('_')[0] for key in self.roi_dict]
        times = keys.count(key_count)

        return key_count + '_' + str(times+1)


    # Check if a new point was created in the map or not
    def check_overlap(self, x, y):
        overlap = 0
        current_point = 0 
        
        if len(self.roi_dict) > 0:
            for key, values in self.roi_dict.items():
                for point in values['pts']:
                    # Check if a new point was created in the map or not
                    if pow(x - point[0], 2) + pow(y - point[1], 2) < pow(RADIUS, 2):
                        overlap =  1
                        self.current_key = key
                        current_point = values['pts'].index(point)
                        break
            
        return overlap, current_point
  
        
    def check_on_a_line(self, x, y):
        on_line = 0
        current_point = 0

        if self.roi_dict:
            for key, points in self.roi_dict.items():
                if key != POINT:
                    distances = [self.distance_point2line(points['pts'][i], points['pts'][(i + 1) % len(points['pts'])], 
                                                          (x, y)) for i in range(len(points['pts']))]
                    if any(distance < THRESHOLD for distance in distances):
                        on_line = 1
                        current_point = distances.index(min(distances))
                        self.current_key = key

        return on_line, current_point


    def distance_two_point(self, pt1, pt2):
        return math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))

    
    def distance_point2line(self, pt1, pt2, pt_check):
        return self.distance_two_point(pt1, pt_check) + self.distance_two_point(pt2, pt_check) - self.distance_two_point(pt1, pt2)


    # Save all drew points as  josn format 
    def save_points(self):

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        file_name = self.img_load_path.split('/')[-1][:-4] + '.json'
        path_file = os.path.join(self.result_folder, file_name)
 
        self.resize_roi(1 / self.scale)
        self.scale = 1

        json_object = json.dumps(self.roi_dict, indent=4)
        # Writing to sample.json
        with open(path_file, "w") as outfile:
            outfile.write(json_object)


    def load_roi(self):
        data = {}
        file_name = self.img_load_path.split('/')[-1][:-4] + '.json'
        path_file = os.path.join(self.result_folder, file_name)

        # a dictionary
        if os.path.exists(path_file):
            f = open(path_file)
            data = json.load(f)

        return data


    def random_color(self):
        r = randrange(255)
        g = randrange(255)
        b = randrange(255)
        rand_color = (r, g, b)
        return rand_color



if __name__ == "__main__":
    # Add arguments to the parser
    parser = argparse.ArgumentParser(description='ROI')
    parser.add_argument('--org', type=str, help='Image path to load')
    parser.add_argument('--des', type=str, help='Image path to save')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    img_path_load = args.org
    img_path_save = args.des

    pd = PolygonDrawer(img_path_load, img_path_save)
    image = pd.roi_process()