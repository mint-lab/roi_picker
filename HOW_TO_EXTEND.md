## ROI Picker Extension Examples

### How to Support Your File Format
You can extend `ROIPicker` to support your own file format. The file input/output is fully done by two member functions as follows.

  * `ROIPicker.read_roi_file(self, filename)`: Reading `filename` and return its _ROI data_
  * `ROIPicker.write_roi_file(self, filename, roi_data)`: Writing `roi_data` to `filename`

:memo: Note) Please remember that the _ROI data_ (and return value `roi_data`) need to contains necessary properties of each ROI. Each ROI needs to have following properties at least.

  * `id` (type `int`): ROI identifier (e.g. 1)
  * `type` (type: `str`): `points` or `line` or `polygon`
  * `color` (type: `tuple`): RGB or BGR color tuple (e.g. (255, 127, 0))
  * `pts` (type: `list`): A series of points (e.g. [(3, 29), (10, 18)])

In Python, `roi_data` is a list of each ROI properties as like following example.
```python
roi_data = [{'id': 1, 'type': 'points', 'color': (255, 127, 0), 'pts': [(3, 29), (10, 18)]},
            {'id': 2, 'type': 'line', '  color': (0, 127, 255), 'pts': [(10, 27), (5, 12)]}]
```

**To support your file format, you can create your own class by inheriting `ROIPicker` and overriding two functions.** For example, your file is a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file which contains only points as a series of `id, x, y`. A point can have a unique `id`. This means that every ROI consists of a single point.

```python
import numpy as np
from roi_picker import ROIPicker, randcolor

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
        return np.savetxt(filename, id_pts, delimiter=',')

if __name__ == '__main__':
    roi_picker = ROIPickerCSV('example.jpg', 'example.csv')
    roi_picker.run_gui()
```