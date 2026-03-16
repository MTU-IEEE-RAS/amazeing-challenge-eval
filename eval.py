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
    fig, axs = plt.subplots(len(solvers), 2) # one column for path length, one for success rate
    
    # Setup maze and start/goal for testing
    generator = wilsons_generator.WilsonsGenerator()
    maze = generator.generate_maze(config.seed, config.maze_width, config.maze_height) # generate a single maze for all solvers to test on
    start_and_goal = generator.generate_start_and_goal(maze)
    
    # Loop through each solver
    for path, solver in solvers.items():
        print(f"Testing {solver} from {path}")
        solver_instance : Solver = solver()
        solver_instance.solve(maze, start_and_goal[0], start_and_goal[1]) # warm up the solver (if it has any initialization overhead)
        # tester = Tester(generator, solver, verbose=True)
        # results = []
        # for i in range(config.num_trials):
        #     results.append(tester.test((maze, start_and_goal)))
