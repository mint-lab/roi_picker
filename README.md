
# ROI PICKER

This repository is intended for drawing points, polygons, and lines on an image and saving their data for labeling purposes.




## Usage
* Command: `python --org roi_tool.py --des org_path des_path`
  * Where:
    * `org_path`: Path to load images.
    * `des_path`: The name of the folder to save the result.
   

* Usage examples
  ```
  python roi_tool.py --org image/210812_camera400.png --des result
  python roi_tool.py --org image/210812_camera500.png --des result
  ```
## GUI feature
 * Keyboard shortcuts
    * ESC: Terminate this program.
    * CTRL+A: Add a new POLYGON.
    * CTRL+W: Add a new LINELINE.
    * CTRL+Q: Add a new POINT.
    * CTRL+N: Remove all points.
    * CTRL+D: Remove all points for the current ROI.
    * CTRL+E: Scale up.
    * CTRL+R: Scale down.
    * CTRL+X: Save ROI.
    * CTRL+L: Load ROI.

  * Mouse shortcuts
    * DOUBLE CLICK: Add, Insert and Delete a point for the current ROI.
        * Add: If that point doesn't exist yet.
        * Delete: If the point just added and the point already exist almost the same.
        * Insert: If the point just added is on a line segment.
    * DRAG: Move the selected point.
