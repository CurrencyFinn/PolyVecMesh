from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

if __name__ == "__main__":
    vtkFile = r'resources\topview\topview_0_0.vtu' 
    pvm = pvm(vtkFile)
    _, ax = plt.subplots(figsize=(10,10))
    meshLines = pvm.createCollection()
    cellLineCollection = LineCollection(meshLines, linewidths=0.5)
    ax.add_collection(cellLineCollection, autolim=True)
    ax.autoscale(enable=True, tight=True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    plt.savefig(r"imgs/coilTop.svg", dpi=300)
