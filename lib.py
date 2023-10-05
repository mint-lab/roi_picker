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