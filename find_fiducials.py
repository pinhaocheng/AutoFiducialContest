from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor
import vtk
import numpy as np
import os
import argparse


def find_fiducials(mesh: vtk.vtkPolyData) -> Fiducials:
    """
    !!!YOUR CODE GOES HERE!!!

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
    # This is a placeholder for your actual logic to find fiducials.
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
    fiducials = Fiducials(color=[0, 1, 0])
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


if __name__ == "__main__":
    # This script is used to find fiducials in a mesh.
    # It takes a scan ID as input and loads the corresponding mesh from a file.
    # The script uses VTK to render the mesh and the fiducials in a 3D window if requested.
    # The script can be run from the command line with the following arguments:
    #   python find_fiducials.py <scan_id> [--display] [--save] [--reference] [--dataset <dataset_name>] [--point-size <size>]
    #
    # Required Arguments:
    #   scan_id: The ID of the scan to process (e.g., 0).
    # Flags:
    #   -d, --display/--no-display: Display the fiducials in a VTK window. Default is not to display.
    #   -s, --save/--no-save: Save the fiducials to a file. Default is not to save.
    #   --reference/--no-reference: Load the reference fiducials from a file for display.
    # Options:
    #   --dataset: The name of the dataset (e.g., training). If you add additional data to the data folder, you can specify the dataset name here.
    #   --point-size: The size of the points in the VTK window. Default is 0.01.

    parser = argparse.ArgumentParser(description="Find fiducials in a mesh.")
    parser.add_argument("id", type=str, help="Scan ID (e.g., 0).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="training",
        help="Dataset name (e.g., training).",
    )
    parser.add_argument(
        "-d",
        "--display",
        action=argparse.BooleanOptionalAction,
        help="Display the fiducials.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action=argparse.BooleanOptionalAction,
        help="Save the fiducials to a file.",
    )
    parser.add_argument(
        "--reference",
        action=argparse.BooleanOptionalAction,
        help="Load the reference fiducials from a file for display.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.01,
        help="Size of the points in the VTK window.",
    )
    args = parser.parse_args()
    scan_id = int(args.id)
    dataset = args.dataset
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data", dataset)
    mesh_path = os.path.join(data_dir, "input_meshes", f"scan_{scan_id:03d}.obj")
    mesh = read_as_vtkpolydata(mesh_path)
    points = find_fiducials(mesh)
    if args.save:
        points.to_file(
            os.path.join(data_dir, "output_points", f"fiducials_{scan_id:03d}.mrk.json")
        )
    else:
        points.print()
    if args.display:
        if args.reference:
            reference_fiducials = Fiducials.from_file(
                os.path.join(
                    data_dir, "reference_points", f"fiducials_{scan_id:03d}.mrk.json"
                )
            )
        else:
            reference_fiducials = None
        renderWindow = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(get_mesh_actor(mesh))
        cp_actors = points.get_actors(size=args.point_size)
        for cp in cp_actors:
            renderer.AddActor(cp)
        if reference_fiducials:
            cp_actors = reference_fiducials.get_actors(size=args.point_size)
            for cp in cp_actors:
                renderer.AddActor(cp)
        renderWindow.Render()
        renderWindowInteractor.Start()
