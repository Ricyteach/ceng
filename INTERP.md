# ceng Interpolation Module

I have not been satisfied with existing tools I have found for engineering interpolation from tables, charts, and 
figures typically found in engineering standards, which require quite a bit of flexible interpolation.

This module is my take on what such tools should look like. The primary application is interpolation of various
standards from engineering building and design code. However, I have other 
applications in mind (such as visual display of interpolation in charts and tables).

```python
# ceng Interpolation Example

from ceng.interp import interp_dict

rows = ("A", "B")
subrows = ("1", "2")
x = [1, 2]
y = [10, 20, 30]
z = {
# TABLE HEADINGS
    #               _X=1_    |    _X=2_
    #         Y=  10 20  30  | 10  20  30
# TABLE ROWS
    ("A", "1"): [[ 1, 2,  3],  [4,  5,  6]],
    ("A", "2"): [[ 2, 4,  6],  [8, 10, 12]],
    ("B", "1"): [[ 3, 6,  9], [12, 15, 18]],
    ("B", "2"): [[ 4, 8, 12], [16, 20, 24]],
}

# create a dictionary of interpolation functions ("interpolants"):
xy_interp = interp_dict(x=x, y=y, z=z)

# calculate an interpolation result:
result =xy_interp[("B", "1")](1.5, 25)
assert result == 12.0
```
