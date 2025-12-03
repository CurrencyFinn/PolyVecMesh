import vtk

# Path to your VTU file
file_path = r"resources\conductor_coil\conductor_coil_0_0.vtu"

# Create a reader for .vtu files
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(file_path)
reader.Update()

# Get the output (vtkUnstructuredGrid)
unstructured_grid = reader.GetOutput()

# Setup mapper and actor
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(unstructured_grid)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().EdgeVisibilityOn()  
actor.GetProperty().SetEdgeColor(1, 1, 1)  
actor.GetProperty().SetColor(0.8, 0.3, 0.3)


# Setup renderer, render window, and interactor
renderer = vtk.vtkRenderer()

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Add actor and start visualization
renderer.AddActor(actor)

render_window.Render()

interactor.Start()