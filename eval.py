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
    fig, axs = plt.subplots(1, 2) # one column for path length, one for success rate
    
    # Get axes
    cost_ax : plt.Axes = axs[0]
    success_ax : plt.Axes = axs[1]
    
    # Setup maze and start/goal for testing
    generator = wilsons_generator.WilsonsGenerator()
    maze = generator.generate_maze(config.seed, config.maze_width, config.maze_height) # generate a single maze for all solvers to test on
    start_and_goal = generator.generate_start_and_goal(maze)
    
    # Loop through each solver
    for path, solver in solvers.items():
        print(f"Testing {solver} from {path}")
        tester = Tester(None, solver(), verbose=True)
        results = tester.test((maze, start_and_goal))
        
        cost_ax.plot([result['time_elapsed'] for result in results if result['is_valid_solution']], 
                     [result['path_length'] for result in results if result['is_valid_solution']], 
                     label=solver.__name__)
        success_ax.plot([result['time_elapsed'] for result in results], 
                        [1 if result['is_valid_solution'] else 0 for result in results], 
                        label=solver.__name__)
    
    # Set titles and labels
    cost_ax.set_title("Path Length vs Time Elapsed")
    cost_ax.set_xlabel("Time Elapsed (s)")
    cost_ax.set_ylabel("Path Length")
    success_ax.set_title("Success Rate vs Time Elapsed")
    success_ax.set_xlabel("Time Elapsed (s)")
    success_ax.set_ylabel("Success Rate")
    
    # Setup legend
    plt.subplots_adjust(bottom=0.2, wspace=0.4)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.15), fancybox=True, shadow=True, ncols=3)
    plt.show()