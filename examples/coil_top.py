from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

if __name__ == "__main__":
    vtkFile = r'resources\topview\topview_0_0.vtu' 
    pvm = pvm(vtkFile)

    meshLines = pvm.createCollection()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax1, ax2 = axes

    cellLineCollection = LineCollection(meshLines, linewidths=0.5, edgecolors="k")
    ax1.add_collection(cellLineCollection)
    ax1.autoscale(enable=True, tight=True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("LineCollection")

    poly = PolyCollection(
        meshLines,
        closed=True,            
        facecolors='white',
        edgecolors='k',
        linewidths=0.5,
        antialiased=False
    )
    ax2.add_collection(poly)
    ax2.autoscale(enable=True, tight=True)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("PolyCollection")

    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(10,10))
    poly = PolyCollection(
        meshLines,
        closed=True,            
        facecolors='white',
        edgecolors='k',
        linewidths=0.5,
        antialiased=False
    )
    ax.add_collection(poly)
    ax.autoscale(enable=True, tight=True)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(r"imgs/coilTop.svg", dpi=300)
    plt.show()
    

    
