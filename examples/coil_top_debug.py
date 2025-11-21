from PolyVecMesh_UPDATE import PolyVecMesh as pvm

if __name__ == "__main__":
    vtkFile = r'resources\topview\topview_0_0.vtu' 
    pvm = pvm(vtkFile)
    pvm.debug_plot()
