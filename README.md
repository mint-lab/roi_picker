
## ROI Picker
_ROI Picker_ is a simple tool to visualize [region-of-interest](https://en.wikipedia.org/wiki/Region_of_interest)s (ROIs) and edit them. It is common for many computer vision tasks to define or select ROIs on an image. _ROI picker_ can provide simple GUI to add, modify, and remove ROIs without complex packages or libraries. _ROI Picker_ is written with [Python](https://www.python.org/) and [OpenCV](https://opencv.org/), and it works with a single Python file, `roi_picker.py`.

_ROI Picker_ now supports the following ROI shapes. Of course, users can extend more through class inheritance or modification.

* A set of points
* A set of line segments
* A polygon

The above shapes are all represented as `[(x1, y1), (x2, y2), ..., (xn, yn)]`.



### How to Run ROI Picker
* **Prerequisite**
  * If you don't install OpenCV, please install OpenCV: `pip install opencv-python`.
* **Command-line usage**
  * `python roi_picker.py image_file [-r roi_file.json] [-c config_file.json]`
    * `-r` (or `--roi_file`): Specify a ROI file which contains ROI data (default: `image_file.json`)
    * `-c` (or `--config_file`): Specify a configuration file which can change visualization and GUI interface (default: `roi_picker.json`)
    * :memo: Note) If a default file does not exist, _ROI Picker_ just starts with empty ROI or with its initial configuration.
* **Demo examples**
  * `python roi_picker.py demo/miraehall_satellite.png`
    * Start _ROI Picker_ with the default ROI file (`demo/miraehall_satellite.json`) and default configuration file (`roi_picker.json`)
  * `python roi_picker.py demo/miraehall_220722.png -r demo/miraehall_camera.json`
    * Start _ROI Picker_ with the specific ROI file (`demo/miraehall_image.json`)
  * `python roi_picker.py demo/miraehall_satellite.png -c demo/bold_style.json`
    * Start _ROI Picker_ with its default ROI file with specific configuration file (`demo/bold_style.json`)
    * :memo: Note) You can customize visualization and keyboard shortcuts by changing a configuration file. Please refer `ROIPicker::get_default_config()` in `roi_picker.py` file.
* **[Extension examples](https://github.com/mint-lab/roi_picker/blob/master/HOW_TO_EXTEND.md)**



### GUI Interface
:memo: Note) _ROI Picker_ provides minimal GUI without any menu or button for simplicity. Its GUI inputs are totally based on mouse and keyboard.

* **Mouse actions**
  * _Click_: Select a ROI
    * If you _click_ at an existing point, the ROI which contains the point is selected.
  * _Double Click_: Add or delete a point
    * If you _double click_ at an existing point, the point will be removed.
    * If you _double click_ on an existing line (for line segment or polygon), a new point will be inserted on the line.
    * If you _double click_ on an image, a new point will be added.
  * `Ctrl`+_Drag_: Move the clicked point

* **Keyboard shortcuts**
  * `ESC`: Terminate _ROI Picker_
  * `Tab`: Select the next ROI
  * `Ctrl`+`P`: Add a new set of **p**oints
  * `Ctrl`+`L`: Add a new set of **l**ine segment
  * `Ctrl`+`G`: Add a new poly**g**on
  * `Ctrl`+`R`: **R**enew _ROI Picker_ (Clear all ROIs)
  * `Ctrl`+`D`: **D**elete the selected ROI
  * `Ctrl`+`M`: I**m**port ROI data from the ROI file
  * `Ctrl`+`E`: **E**xport ROI data to the ROI file
  * `Ctrl`+`F`: Export con**f**iguration to a JSON file
  * `Ctrl`+`Z`: Show and hide the image **z**oom
  * `Ctrl`+`T`: Show and hide the s**t**atus of the selected ROI
  * `+`: Zoom up the image
  * `-`: Zoom down the image



### Screenshots
You can see the following demo if you run the above three _usage examples_, respectively.

  <img width="452px" src="https://github.com/mint-lab/roi_picker/blob/master/demo/miraehall_satellite_demo.png" />

  <img width="897px" src="https://github.com/mint-lab/roi_picker/blob/master/demo/miraehall_220722_demo.png" />

:memo: Note) Please remember that you can customize visualization and keyboard shortcuts when you apply your own configuration file (e.g. `demo/bold_style.json`). The following example is same data only with a different configuration file.

  <img width="542px" src="https://github.com/mint-lab/roi_picker/blob/master/demo/miraehall_satellite_demo_bold.png" />



### Authors
* [Nguyen Cong Quy](https://github.com/ncquy)
* [Sunglok Choi](https://mint-lab.github.io/sunglok/)