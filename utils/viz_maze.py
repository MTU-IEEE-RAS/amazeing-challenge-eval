import numpy as np
from typing import List, Tuple
from amazeing_challenge.contracts import Maze


def create_maze_array(maze: Maze, num_rows: int, num_cols: int) -> np.ndarray:
    maze_array = np.zeros((2*num_rows + 1, 2*num_cols + 1)) # Add 2 for border
    for i in range(num_rows): 
        for j in range(num_cols):
            # Create offset indices due to the border
            i_maze = 2*i + 1
            j_maze = 2*j + 1
            
            # Every other row/col is a feasible cell
            maze_array[i_maze, j_maze] = 255 
            
            # Check for wall between columns
            if j + 1 < num_cols:
                if not maze.has_edge((i, j), (i, j+1)):
                    maze_array[i_maze, j_maze+1] = 0 # Wall between cells in column
                else:
                    maze_array[i_maze, j_maze+1] = 255 # Corridor between cells along row
                
            # Check for wall between rows
            if i + 1 < num_rows:
                if not maze.has_edge((i, j), (i+1, j)):
                    maze_array[i_maze+1, j_maze] = 0 # Wall between cells in row
                else:
                    maze_array[i_maze+1, j_maze] = 255 # Corridor between cells along column
    return maze_array


def add_path_to_maze(path: List[Tuple[int, int]]) -> np.ndarray:
    pass