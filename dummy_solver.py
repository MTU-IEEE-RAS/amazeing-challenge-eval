from amazeing_challenge.contracts import Solver, Maze

class DummySolver(Solver):

    def solve(self, maze : Maze, start, goal) -> list:
        return []