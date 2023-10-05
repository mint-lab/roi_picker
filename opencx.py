'''
OpenCX: Sunglok's OpenCV Extension

OpenCX aims to provide extended functionality and tools to OpenCV for more convenience.
It consists of several header files, 'opencx.hpp' and others, which only depend on OpenCV in C++.
Just include the file to your project. It will work without complex configuration and dependency.
OpenCX is Beerware so that it is free to use and distribute.

- Homepage: https://github.com/mint-lab/opencx

@author  Sunglok Choi (https://mint-lab.github.io/sunglok)
'''

import cv2 as cv
import numpy as np

class VideoWriter:
    '''A video writer necessary for less number of parameters'''

    def __init__(self, fname='', fourcc='FMP4', fps=10):
        '''A constructor'''
        self.video = cv.VideoWriter()
        self.open(fname, fourcc, fps)

    def __del__(self):
        '''A destructor'''
        self.release()

    def open(self, fname, fourcc='FMP4', fps=10):
        '''Configure a video file'''
        if self.video.isOpened():
            self.video.release()
        self.fname = fname
        self.fourcc = fourcc.upper()
        self.fps = fps
        return True

    def isOpened(self):
        '''Check whether the video is configured or not'''
        return len(self.fname) > 0

    def write(self, img):
        '''Write a image to the video and open the video if necessary'''
        if len(img.shape) < 2 or not self.fname:
            return False
        if not self.video.isOpened():
            fourcc = cv.VideoWriter_fourcc(*self.fourcc)
            if not self.video.open(self.fname, fourcc, self.fps, (img.shape[1], img.shape[0]), len(img.shape) > 2):
                return False
        if self.video.isOpened():
            self.video.write(img)
            return True
        return False

    def release(self):
        '''Close the video'''
        if self.video.isOpened():
            self.video.release()
            self.fname = ''
            return True
        return False

def putText(img, text, org_tl, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1, colorOutline=(0, 0, 0), thicknessOutline=2, lineSpacing=1.5):
    '''Draw a multi-line text with outline'''
    assert isinstance(text, str)

    org_tl = np.array(org_tl, dtype=float)
    assert org_tl.shape == (2,)

    for line in text.splitlines():
        (_, h), _ = cv.getTextSize(text=line, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
        org = tuple((org_tl + [0, h]).astype(int))

        if colorOutline is not None:
            cv.putText(img, text=line, org=org, fontFace=fontFace, fontScale=fontScale, color=colorOutline, thickness=thickness*thicknessOutline, lineType=cv.LINE_AA)
        cv.putText(img, text=line, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness, lineType=cv.LINE_AA)
        org_tl += [0, h * lineSpacing]



if __name__ == '__main__':
    '''Test modules and functions'''

    # Test 'VideoWriter'
    video = VideoWriter('utils_test.avi', 'FMP4')
    assert video.isOpened()
    img = np.zeros((240, 320, 3), dtype='uint8')
    img[:,:,1] = 255
    for i in range(10):
        assert video.write(img)
    assert video.release()
