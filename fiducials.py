from dataclasses import dataclass, field
from typing import List
from types import MappingProxyType
import json
import vtk
import os

CONTROL_POINT_TEMPLATE = MappingProxyType(
    {
        "id": "",
        "label": "",
        "description": "",
        "associatedNodeID": "",
        "position": [],
        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
        "selected": True,
        "locked": False,
        "visibility": True,
        "positionStatus": "defined",
    }
)

SCHEMA = "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"

MARKUP_TEMPLATE = MappingProxyType(
    {
        "type": "Fiducial",
        "coordinateSystem": "LPS",
        "coordinateUnits": "mm",
        "locked": False,
        "fixedNumberOfControlPoints": False,
        "labelFormat": "%N-%d",
        "lastUsedControlPointNumber": 0,
        "controlPoints": [],
        "measurements": [],
        "display": {},
    }
)

DISPLAY_TEMPLATE = MappingProxyType(
    {
        "visibility": True,
        "opacity": 1.0,
        "color": [0.4, 1.0, 1.0],
        "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
        "activeColor": [0.4, 1.0, 0.0],
        "propertiesLabelVisibility": False,
        "pointLabelsVisibility": True,
        "textScale": 3.0,
        "glyphType": "Sphere3D",
        "glyphScale": 3.0,
        "glyphSize": 5.0,
        "useGlyphScale": True,
        "sliceProjection": False,
        "sliceProjectionUseFiducialColor": True,
        "sliceProjectionOutlinedBehindSlicePlane": False,
        "sliceProjectionColor": [1.0, 1.0, 1.0],
        "sliceProjectionOpacity": 0.6,
        "lineThickness": 0.2,
        "lineColorFadingStart": 1.0,
        "lineColorFadingEnd": 10.0,
        "lineColorFadingSaturation": 1.0,
        "lineColorFadingHueOffset": 0.0,
        "handlesInteractive": False,
        "translationHandleVisibility": True,
        "rotationHandleVisibility": True,
        "scaleHandleVisibility": False,
        "interactionHandleScale": 3.0,
        "snapMode": "toVisibleSurface",
    }
)


@dataclass
class ControlPoint:
    position: List[float]
    label: str = ""
    id: int = 0
    description: str = ""

    def __post_init__(self):
        self.position = [float(coord) for coord in self.position]
        if len(self.position) != 3:
            raise ValueError("Position must be a list of three floats")

    def to_dict(self) -> dict:
        d = CONTROL_POINT_TEMPLATE.copy()
        d.update(
            {
                "id": f"{self.id}",
                "label": self.label,
                "description": self.description,
                "position": self.position,
            }
        )
        return d

    @staticmethod
    def from_dict(d: dict) -> "ControlPoint":
        return ControlPoint(
            position=d["position"],
            label=d["label"],
            id=int(d["id"]),
            description=d["description"],
        )


@dataclass
class Fiducials:
    control_points: List[ControlPoint] = field(default_factory=list)
    color: List[float] = field(default_factory=lambda: [0.4, 1.0, 1.0])
    coordinate_system: str = "LPS"

    def to_dict(self) -> dict:
        d = MARKUP_TEMPLATE.copy()
        d.update(
            {
                "coordinateSystem": self.coordinate_system,
                "controlPoints": [cp.to_dict() for cp in self.control_points],
            }
        )
        d["display"] = DISPLAY_TEMPLATE.copy()
        d["display"].update({"color": self.color})
        return {"@schema": SCHEMA, "markups": [d]}

    def to_file(self, filename: str) -> None:
        file_dir = os.path.dirname(filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def set_coordinate_system(self, coordinate_system: str) -> None:
        if coordinate_system not in ["LPS", "RAS"]:
            raise ValueError("Coordinate system must be either 'LPS' or 'RAS'")
        if self.coordinate_system != coordinate_system:
            for cp in self.control_points:
                cp.position = [-cp.position[0], -cp.position[1], cp.position[2]]
            self.coordinate_system = coordinate_system

    def get_actors(self, size=1) -> List[vtk.vtkActor]:
        """Get vtk actors for the control points."""
        actors = []
        for cp in self.control_points:
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(cp.position)
            sphere_source.SetRadius(size)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere_source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.color)
            actors.append(actor)
        return actors

    def print(self) -> None:
        """Print the control points."""
        id_max_length = max(2, max(len(str(cp.id)) for cp in self.control_points))
        label_max_length = max(len(cp.label) for cp in self.control_points)
        id_format = f"{{:<{id_max_length}}}"
        label_format = f"{{:<{label_max_length}}}"
        print(f"{id_format.format('ID')} | {label_format.format('Label')} | Position")
        print("-" * (id_max_length + label_max_length + 28))
        for cp in self.control_points:
            pos_str = ", ".join(f"{coord: .3f}" for coord in cp.position)
            print(
                f"{id_format.format(cp.id)} | {label_format.format(cp.label)} | {pos_str}"
            )

    @staticmethod
    def from_dict(data: dict) -> "Fiducials":
        control_points = [
            ControlPoint(
                position=cp["position"],
                label=cp["label"],
                id=cp["id"],
                description=cp["description"],
            )
            for cp in data["markups"][0]["controlPoints"]
        ]
        color = data["markups"][0]["display"]["color"]
        coordinate_system = data["markups"][0]["coordinateSystem"]
        return Fiducials(
            control_points=control_points,
            color=color,
            coordinate_system=coordinate_system,
        )

    @staticmethod
    def from_file(filename: str) -> "Fiducials":
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        with open(filename, "r") as file:
            data = json.load(file)
        return Fiducials.from_dict(data)
