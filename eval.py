import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Iterable, Callable, Any
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pydantic import BaseModel, Field
from yaml import safe_load
import inspect
import numpy as np

from utils.viz_maze import create_maze_array

from amazeing_challenge.contracts import Solver
from amazeing_challenge.generators import wilsons_generator
from amazeing_challenge.tester import Tester


def load_child_solver_from_file(path: pathlib.Path) -> type:
    """
    Import a module from *path* and return its attribute *func_name*.
    Raises ImportError/AttributeError on failure.
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {path}")
    module = importlib.util.module_from_spec(spec)  # type: ModuleType
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore
    
    solver_subclass : type = None
    for attr_name in module.__dict__.keys():
        attr = getattr(module, attr_name)
        if inspect.isclass(attr) and issubclass(attr, Solver):
            solver_subclass = attr
            
    if solver_subclass is None:
        raise ImportError(f"No Solver subclass found in {path}")
    return solver_subclass


def load_child_solvers_from_directory(
    directory: pathlib.Path,
    blacklist: Iterable[pathlib.Path]
) -> dict[pathlib.Path, type]:
    """
    Scan *directory* for .py files and import *func_name* from each.
    Returns a mapping from file path to the loaded callable.
    """
    results: dict[pathlib.Path, type] = {}
    for py in directory.iterdir():
        if py not in blacklist and py.suffix == ".py" and py.is_file():
            results[py] = load_child_solver_from_file(py)
    return results


class EvalConfig(BaseModel):
    seed: int = Field(default=0, ge=0)
    maze_width: int = Field(default=5, ge=1)
    maze_height: int = Field(default=5, ge=1)
    num_trials: int = Field(default=100, ge=1)
    additional_edges: int = Field(default=0, ge=0)


# example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", default="/home/mtu_ieee_ras_ws/dev/amazeing-challenge-eval/config/eval.yaml", help="Configuration file for evaluation")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = safe_load(f)
    config = EvalConfig(**config_data)

    base = pathlib.Path(".")
    blacklist = [pathlib.Path("eval.py")]
    solvers = load_child_solvers_from_directory(base, blacklist)
    stats_fig, stats_axs = plt.subplots(1, 2) # one column for path length, one for success rate
    
    # Get axes for stat visualization
    cost_ax : plt.Axes = stats_axs[0]
    success_ax : plt.Axes = stats_axs[1]
    
    # Setup maze and start/goal for testing
    generator = wilsons_generator.WilsonsGenerator()
    maze = generator.generate_maze(config.seed, config.maze_width, config.maze_height, config.additional_edges) # generate a single maze for all solvers to test on
    start_and_goal = generator.generate_start_and_goal(maze)
    
    # Setup Maze Visualization
    maze_fig, maze_ax = plt.subplots(1,1)
    maze_array = create_maze_array(maze, config.maze_height, config.maze_width)
    maze_ax.imshow(maze_array, cmap='gray')
    maze_ax.set_xticks(np.arange(1, 2*config.maze_width, 2))
    maze_ax.set_yticks(np.arange(1, 2*config.maze_height, 2))
    maze_ax.set_xticklabels(np.arange(0, config.maze_width, 1))
    maze_ax.set_yticklabels(np.arange(0, config.maze_height, 1))
    maze_ax.scatter(2*start_and_goal[0][1]+1, 2*start_and_goal[0][0]+1, color='green', s=20, label='Start')
    maze_ax.scatter(2*start_and_goal[1][1]+1, 2*start_and_goal[1][0]+1, color='red', s=20, label='Goal')
    
    if start_and_goal[0] == start_and_goal[1]:
        raise ValueError("Start and goal are the same, cannot evaluate solvers on this maze. Choose a different seed!")
    
    # Loop through each solver
    for path, solver in solvers.items():
        print(f"Testing {solver} from {path}")
        tester = Tester(None, solver(), verbose=False)
        results = tester.test((maze, start_and_goal))
        
        cost_ax.plot([result['time_elapsed'] for result in results if result['is_valid_solution']], 
                     [result['path_length'] for result in results if result['is_valid_solution']], 
                     label=solver.__name__)
        success_ax.plot([result['time_elapsed'] for result in results], 
                        [1 if result['is_valid_solution'] else 0 for result in results], 
                        label=solver.__name__)
        
        path = results[-1]['solution']
        x = [2*p[1]+1 for p in path[0]]
        y = [2*p[0]+1 for p in path[0]]
        maze_ax.plot(x, y, color='blue', linewidth=2, label=solver.__name__)
    
    # Set titles and labels for stats fig
    cost_ax.set_title("Path Length vs Time Elapsed")
    cost_ax.set_xlabel("Time Elapsed (s)")
    cost_ax.set_ylabel("Path Length")
    success_ax.set_title("Success Rate vs Time Elapsed")
    success_ax.set_xlabel("Time Elapsed (s)")
    success_ax.set_ylabel("Success Rate")
    
    # Setup legend for stats fig
    stats_fig.subplots_adjust(bottom=0.2, wspace=0.4)
    stats_fig.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.15), fancybox=True, shadow=True, ncols=3)
    
    maze_fig.legend()
    
    plt.show()