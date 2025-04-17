from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor
import vtk
import numpy as np

reference_fiducials = Fiducials.from_file(
    R"data\training\output_pts\fiducials_000.mrk.json"
)
mesh = read_as_vtkpolydata(R"data\training\input_meshes\scan_000.obj")


def find_fiducials(mesh: vtk.vtkPolyData) -> Fiducials:
    """
    Find fiducials in a mesh.

    Input:
        mesh: vtkPolyData object representing the mesh.

    Output:
        fiducials: Fiducials object containing the found fiducials.

    This example is highly reductive and inaccurate, but demonstrates the flow of data through the function
    """
    # get the positions of the mesh vertices
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    arr = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        p = points.GetPoint(i)
        arr[i, 0] = p[0]
        arr[i, 1] = p[1]
        arr[i, 2] = p[2]

    # Analyze the mesh vertices to find fiducials
    # This is a placeholder for the actual logic to find fiducials.
    center = np.mean(arr, axis=0)
    pmax = np.max(arr, axis=0)
    pmin = np.min(arr, axis=0)
    size = pmax - pmin
    ravg = np.mean(size)
    arr_centered = arr - center
    r_lp = np.sqrt(arr_centered[:, 0] ** 2 + arr_centered[:, 1] ** 2)
    r_ps = np.sqrt(arr_centered[:, 1] ** 2 + arr_centered[:, 2] ** 2)
    ps = arr[r_lp < ravg / 30, :]
    pl = arr[r_ps < ravg / 30, :]
    nasion = ps[np.argmax(ps[:, 2]), :]
    nasion_r = np.sqrt(np.sum((nasion - center) ** 2))
    left_ear = pl[np.argmax(pl[:, 0]), :]
    left_ear_r = np.sqrt(np.sum((left_ear - center) ** 2))
    right_ear = pl[np.argmin(pl[:, 0]), :]
    right_ear_r = np.sqrt(np.sum((right_ear - center) ** 2))
    left_eye_outside = left_ear * 0.5 + nasion * 0.5
    left_eye_outside_r = np.sqrt(np.sum((left_eye_outside - center) ** 2))
    left_eye_outside = (
        (left_eye_outside - center)
        * (left_ear_r * 0.5 + nasion_r * 0.5)
        / left_eye_outside_r
    ) + center
    left_eye_inside = left_ear * 0.25 + nasion * 0.75
    left_eye_inside_r = np.sqrt(np.sum((left_eye_inside - center) ** 2))
    left_eye_inside = (
        (left_eye_inside - center)
        * (left_ear_r * 0.25 + nasion_r * 0.75)
        / left_eye_inside_r
    ) + center
    right_eye_outside = right_ear * 0.5 + nasion * 0.5
    right_eye_outside_r = np.sqrt(np.sum((right_eye_outside - center) ** 2))
    right_eye_outside = (
        (right_eye_outside - center)
        * (right_ear_r * 0.5 + nasion_r * 0.5)
        / right_eye_outside_r
    ) + center
    right_eye_inside = right_ear * 0.25 + nasion * 0.75
    right_eye_inside_r = np.sqrt(np.sum((right_eye_inside - center) ** 2))
    right_eye_inside = (
        (right_eye_inside - center)
        * (right_ear_r * 0.25 + nasion_r * 0.75)
        / right_eye_inside_r
    ) + center

    # Create the Fiducials object and populate the control points
    fiducials = Fiducials()
    fiducials.control_points.append(ControlPoint(left_ear, "left_ear"))
    fiducials.control_points.append(ControlPoint(left_eye_outside, "left_eye_outside"))
    fiducials.control_points.append(ControlPoint(left_eye_inside, "left_eye_inside"))
    fiducials.control_points.append(ControlPoint(nasion, "nasion"))
    fiducials.control_points.append(ControlPoint(right_eye_inside, "right_eye_inside"))
    fiducials.control_points.append(
        ControlPoint(right_eye_outside, "right_eye_outside")
    )
    fiducials.control_points.append(ControlPoint(right_ear, "right_ear"))

    return fiducials


fiducials = find_fiducials(mesh)

renderWindow = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderer.AddActor(get_mesh_actor(mesh))
cp_actors = fiducials.get_actors(size=0.01)
for cp in cp_actors:
    renderer.AddActor(cp)
renderWindow.Render()
renderWindowInteractor.Start()
