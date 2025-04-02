import json
import numpy as np
from evogym import get_full_connectivity
from evogym.world import WorldObject
from evogym.utils import sample_robot



def load_robot_from_json(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get the length and width of the grid
    grid_width = data['grid_width']
    grid_height = data['grid_height']

    # Initialize an empty grid matrix
    robot = np.zeros((grid_height, grid_width), dtype=float)

    # Read indices and types, fill in robot matrix
    indices = data['objects']['robot']['indices']
    types = data['objects']['robot']['types']

    for idx, val in zip(indices, types):
        row, col = divmod(idx, grid_width)
        robot[row, col] = val

    robot = np.flipud(robot)

    return robot, get_full_connectivity(robot)
    #robot_object = WorldObject.from_json(file_path)
    #return robot_object.get_structure(), robot_object.get_connections()



if __name__ == "__main__":
    robot_path = ("C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym\\examples\\saved_data\\hand_design_experiment\\generation_49\\structure\\0.npz")
    data = np.load(robot_path)
    for key in data.files:
        print(f"\nContent of {key}:")
        print(data[key])