[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CurrencyFinn/TermSymbolCpp/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-0.0.1.0-blue.svg)](https://github.com/CurrencyFinn/TermSymbolCpp/releases/)

# PolyVecMesh

PolyVecMesh is a lightweight tool for visualising CFD polyhedral and hexahedral meshes in 2D vector format. It converts meshes exported from ParaView (VTK Multi-Block format) into clean, publication-quality (vector format) graphics that can be plotted directly with matplotlib.

This enables high-resolution, scalable mesh figures, something that is not easily achievable in ParaView for vector graphics.

### Requirements
- Mesh exported from ParaView as VTK Multi-Block
- Python ≥3.8

### Python Dependencies
- numpy
- matplotlib
- xml

## Usage

1. In ParaView, create a 2D slice of your mesh.  
2. Export the slice as XML Multi Block Data (`.vtm`) with Data Mode: ASCII.  
3. In Python, load the file using the `PolyVecMesh` class.  
4. Generate the mesh line data using `createCollection()`.  
5. Pass the resulting array into a `matplotlib.collections.LineCollection` for plotting.

Example (see the `examples` folder for full scripts):
```python
from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

if __name__ == "__main__":
    vtkFile = r"resources\topview\topview_0_0.vtu"
    pvm = pvm(vtkFile)

    _, ax = plt.subplots(figsize=(10, 10))
    meshLines = pvm.createCollection()

    cellLineCollection = LineCollection(meshLines, linewidths=0.5)
    ax.add_collection(cellLineCollection, autolim=True)

    ax.autoscale(enable=True, tight=True)
    ax.set_aspect("equal", adjustable="box")

    plt.show()
```

