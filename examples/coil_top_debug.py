from PolyVecMesh import PolyVecMesh as pvm

if __name__ == "__main__":
    vtmFile = r'resources\topview.vtu' 
    pvm = pvm(vtmFile)
    pvm.debug_plot()
