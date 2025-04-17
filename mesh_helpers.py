import vtk
from pathlib import Path


def read_as_vtkpolydata(file_name):
    suffix_to_reader_dict = {
        ".ply": vtk.vtkPLYReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".obj": vtk.vtkOBJReader,
        ".stl": vtk.vtkSTLReader,
        ".vtk": vtk.vtkPolyDataReader,
        ".g": vtk.vtkBYUReader,
    }
    path = Path(file_name)
    ext = path.suffix.lower()
    if path.suffix not in suffix_to_reader_dict:
        raise ValueError(f"File format {ext} not supported by reader")
    reader = suffix_to_reader_dict[ext]()
    if ext == ".g":
        reader.SetGeometryName(file_name)
    else:
        reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    return poly_data


def convert_between_ras_and_lps(mesh: vtk.vtkPointSet) -> vtk.vtkPointSet:
    """Converts a mesh (polydata, unstructured grid or even just a point cloud) between the LPS (left-posterior-superior) coordinate system and
    RAS (right-anterior-superior) coordinate system."""

    transform_ras_to_lps = vtk.vtkTransform()
    transform_ras_to_lps.Scale(-1, -1, 1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform_ras_to_lps)
    transformFilter.Update()

    return transformFilter.GetOutput()


def get_mesh_actor(mesh):
    """Get vtk actor for a mesh."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    return actor


def preview_mesh(mesh):
    actor = get_mesh_actor(mesh)
    renderWindow = vtk.vtkRenderWindow()
    renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    renderWindow.Render()
    renderWindowInteractor.Start()
