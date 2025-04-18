from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor
import vtk
import numpy as np
import os
import argparse

if __name__ == "__main__":
    # This script is used to visualize fiducials in a mesh.
    # It takes a scan ID as input and loads the corresponding mesh and fiducials from files.
    # The script uses VTK to render the mesh and the fiducials in a 3D window.
    # The script can be run from the command line with the following arguments:
    #   python show_fiducials.py <scan_id> [--output] [--reference] [--dataset <dataset_name>] [--point-size <size>]
    #
    # Required Arguments:
    #   scan_id: The ID of the scan to process (e.g., 0).
    # Flags:
    #   --output/--no-output: Load the output fiducials from a file for display.
    #   --reference/--no-reference: Load the reference fiducials from a file for display.
    # Options:
    #   --dataset: The name of the dataset (e.g., training). If you add additional data to the data folder, you can specify the dataset name here.
    #   --point-size: The size of the points in the VTK window. Default is 0.01.

    parser = argparse.ArgumentParser(description="Show fiducials in a mesh.")
    parser.add_argument("id", type=str, help="Scan ID (e.g., 0).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="training",
        help="Dataset name (e.g., training).",
    )
    parser.add_argument(
        "--output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the output fiducials from a file for display.",
    )
    parser.add_argument(
        "--reference",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    if args.output:
        output_fiducials = Fiducials.from_file(
            os.path.join(data_dir, "output_points", f"fiducials_{scan_id:03d}.mrk.json")
        )
    else:
        output_fiducials = None
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
    for pointset in [output_fiducials, reference_fiducials]:
        if pointset:
            cp_actors = pointset.get_actors(size=args.point_size)
            for cp in cp_actors:
                renderer.AddActor(cp)
    renderWindow.Render()
    renderWindowInteractor.Start()
