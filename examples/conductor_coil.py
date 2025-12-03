from PolyVecMesh import PolyVecMesh as pvm
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

if __name__ == "__main__":
    vtmFile = r'resources\conductor_coil.vtm' 
    pvm = pvm(vtmFile)
    
    _, ax = plt.subplots(figsize=(10,10))

    for i, name in enumerate(pvm.names):

        colors = ["#cc3311", "#ee7733", "#0077bb"] # ptol vibrant
        meshLines = pvm.createCollection(name)
    
        poly = PolyCollection(
            meshLines,
            closed=True,            
            facecolors=colors[i],
            edgecolors='k',
            linewidths=0.1,
            antialiased=False
        )
        ax.add_collection(poly)
        
        # cellLineCollection = LineCollection(meshLines, linewidths=0.5, edgecolors=colors[i])
        # ax.add_collection(cellLineCollection)

    ax.autoscale(enable=True, tight=True)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(r"imgs/multi_coil.svg", dpi=300, bbox_inches="tight")
    plt.show()  
    

    
