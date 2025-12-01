from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.collections import PolyCollection
import numpy as np

if __name__ == "__main__":
    vtkFile = r'resources\motorBike.vtm' 
    pvm = pvm(vtkFile, tol=np.inf)
    _, ax = plt.subplots(figsize=(10,10))
    meshLines = pvm.createCollection("internalMesh")
    poly = PolyCollection(meshLines, closed=False, facecolors='white', edgecolors="k", linewidths=0.5, antialiased=True, zorder=-1)
    ax.add_collection(poly, autolim=True)
    
    ax.autoscale(enable=True, tight=True)
    ax.set_aspect('equal', adjustable='box')

    x0, x1 = -0.3, 1.75
    y0, y1 = 0.0, 1.4

    aspectWidthToHeight = (y1 - y0) / (x1 - x0)
    inset_width = 5

    inset = inset_axes(
        ax,
        width=inset_width,
        height=inset_width * aspectWidthToHeight,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    inset.add_collection(PolyCollection(meshLines, closed=False, facecolors='white', edgecolors="k", linewidths=0.5, antialiased=True, zorder=-1))
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.set_aspect("equal", adjustable="box")
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in inset.spines.values():
        spine.set_linewidth(2)

    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black", linewidth=2)
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor="black", linewidth=2, zorder=1)
    ax.add_patch(rect)

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(r"imgs/motorBike.svg", dpi=300, bbox_inches="tight")
    plt.show()
    