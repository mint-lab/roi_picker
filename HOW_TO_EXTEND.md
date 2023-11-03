## ROI Picker Extension Examples

### How to Support Your File Format
You can extend `ROIPicker` to support your own file format. The file input/output is fully done by two member functions as follows.

  * `ROIPicker.read_roi_file(self, filename)`: Reading `filename` and return its ROI data
  * `ROIPicker.write_roi_file(self, filename, roi_data)`: Writing `roi_data` to `filename`

:memo: Note) Please remember that the member variable `roi_data` contains every information about each ROI. Each ROI has properties as follows.

  * `id` (type `int`): ROI identifier (e.g. 1)
  * `type` (type: `str`): `points` or `line` or `polygon`
  * `color` (type: `tuple`): RGB or BGR color tuple (e.g. (255, 127, 0))
  * `pts` (type: `list`): A series of points (e.g. [(3, 29), (10, 18)])

In Python, `roi_data` is a list of each ROI properties as like following example.
```python
roi_data = [{'id': 1, 'type': 'points', 'color': (255, 127, 0), 'pts': [(3, 29), (10, 18)]},
            {'id': 2, 'type': 'line', '  color': (0, 127, 255), 'pts': [(10, 27), (5, 12)]}]
```

**To support your file format, you can create your own class by inheriting `ROIPicker` and overriding two functions.** The following is an example to read points from a CSV file.

```python
import numpy as np
from mint.roi_picker import ROIPicker, randcolor

class ROIPickerCSV(ROIPicker):
    '''ROI Picker with CSV files'''

    def read_roi_file(self, filename):
        '''Read ROI data from the given CSV file `filename`'''
        id_pts = np.loadtxt(filename, delimiter=',')
        roi_data = [{'id': int(id), 'type': 'points', 'color': randcolor(), 'pts': [(x, y)]}
                    for (id, x, y) in id_pts]
        return roi_data

    def write_roi_file(self, filename, roi_data):
        '''Write the given ROI data to the given CSV file `filename`'''
        id_pts = [(roi['id'], pt[0], pt[1]) for roi in roi_data for pt in roi['pts']]
        return np.savetxt(filename, id_pts)
```