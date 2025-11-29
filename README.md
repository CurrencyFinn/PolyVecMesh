[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CurrencyFinn/PolyVecMesh/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-blue.svg)](https://github.com/CurrencyFinn/PolyVecMesh/releases/)

<img src="examples/imgs/logo.svg" alt="Logo" width="300">

PolyVecMesh is a lightweight tool for visualising CFD polyhedral and hexahedral meshes in 2D vector format. It converts meshes exported from ParaView (VTK Multi-Block format) into clean, publication-quality (vector format) graphics that can be plotted directly with matplotlib.

This enables high-resolution, scalable mesh figures, something that is not easily achievable in ParaView for vector graphics.

### Requirements
- Mesh exported from ParaView as VTK Multi-Block
- Python ≥3.8

### Python Dependencies
- numpy
- matplotlib

## Mesh Visualisations

<p align="center"> 
  <img src="examples/imgs/motorBike.svg" alt="Motor Bike Mesh" style="width:67.97%;"> 
  <img src="examples/imgs/coilTop.svg" alt="Coil Top Mesh" style="width:29.03%;"> 
</p>

## Usage


0. Local install, from the root folder (where `pyproject.toml` is):
```bash
pip install -e .
```
1. In ParaView, create a 2D slice of your mesh.  
2. Export the slice as XML Multi Block Data (`.vtm`) with Data Mode: ASCII.  
3. In Python, load the file using the `PolyVecMesh` class.  
4. Generate the mesh line data using `createCollection()`.  
5. Pass the resulting array into a `matplotlib.collections.LineCollection` for plotting.

Example (see the `examples` folder for full scripts):
```python
from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

if __name__ == "__main__":
    vtkFile = r"resources\topview\topview_0_0.vtu"
    pvm = pvm(vtkFile)

    _, ax = plt.subplots(figsize=(10, 10))
    meshLines = pvm.createCollection()

    poly = PolyCollection(meshLines, closed=False, linewidths=0.5,   facecolors='white', edgecolors="k", linewidths=0.5)
    ax.add_collection(poly, autolim=True)

    ax.autoscale(enable=True, tight=True)
    ax.set_aspect("equal", adjustable="box")

    plt.show()
```

## TODO

### Functionality
- [ ] Multi-region support + VTK colouring
- [ ] Smarter detection of cells outside the slice plane when using implicit clipping
    - e.g., pre-filter points with a large deviation from the normal
    - introduce a `maxDistanceOffSlice` threshold
- [ ] Auto-detect `maxDistanceOffSlice` (essentially a custom slicing tool)
- [ ] Include off-normal 2D plane creation, now only a plane can be made in the mesh aligned with the axis
    - e.g., rotate all points until it is aligned with the axis to remove excess points.
- [ ] PyPI installation

### Optimization
- [ ]

### Bugs
- [ ] In `debug_plot`, the `self.colors` list is not updated when unwanted faces are removed Will be updated when including multi-region.

## License

[MIT](https://github.com/CurrencyFinn/PolyVecMesh/blob/main/LICENSE)




















