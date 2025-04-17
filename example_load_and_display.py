from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor
import vtk

reference_fiducials = Fiducials.from_file(
    R"data\training\output_pts\fiducials_001.mrk.json"
)
mesh = read_as_vtkpolydata(R"data\training\input_meshes\scan_001.obj")

renderWindow = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderer.AddActor(get_mesh_actor(mesh))
cp_actors = reference_fiducials.get_actors(size=0.01)
for cp in cp_actors:
    renderer.AddActor(cp)
renderWindow.Render()
renderWindowInteractor.Start()
