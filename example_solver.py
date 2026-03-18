from amazeing_challenge.contracts import Solver, Maze

from utils.graph_search import dijkstras

class ExampleSolver(Solver):

    def solve(self, maze : Maze, start, goal) -> list:
        costCb = lambda pose1, pose2 : 1.0 # All edges are feasible neighbors. Therefore, all edges have a cost of 1.0.
        neighborsCb = lambda pose : [edge[1] for edge in maze.get_edges(pose)] # Neighbors are only the last element of edges
        return dijkstras(start, goal, neighborsCb, costCb, verbose=False)