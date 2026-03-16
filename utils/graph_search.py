import numpy as np
from typing import List, Dict, Tuple, Callable, Iterator
import heapq

# Create some useful types


Pose = Tuple[int, int]
CostCb = Callable[[Pose, Pose], float]
NeighborsCb = Callable[[Pose], List[Pose]]
Path = List[Pose]
Solution = Tuple[Path, float]


class Node:
    def __init__(self, pose: Pose, cost: float = np.inf):
        self.pose = pose
        self.cost_from_start = cost
        self.prev : Pose = None
        self.neighbors : List[Pose] = None

    # This ordering (used in the priority queue) is very important! If we were to include a "heursitic" when ordering, we end up with A*
    def __lt__(self, node) -> bool:
        if isinstance(node, Node):
            return self.cost_from_start < node.cost_from_start
        raise NotImplementedError
    
    def __le__(self, node) -> bool:
        if isinstance(node, Node):
            return self.cost_from_start <= node.cost_from_start
        raise NotImplementedError


Graph = Dict[Pose, Node]


def dijkstras(start_pose: Pose, end_pose: Pose, neighbors_cb: NeighborsCb, cost_cb: CostCb, verbose: bool = False) -> Iterator[Solution]:
    """
    Dijkstras algorithm for finding the shortest path between two nodes in a graph

    Parameters
    ----------
    start_pose : Pose
        Start pose
    end_pose : Pose
        End pose
    graph : Graph
        Graph (start and end must be in the graph!)
    cost_cb : CostCb
        Callback describing the cost between two nodes
    verbose : bool, defaults to False
        Flag to enable extra print statements

    Yields
    ------
    Iterator[Solution]
        Use an iterator to return the current best path
    """

    # Create a graph to store all nodes
    graph : Graph = {start_pose: Node(start_pose, 0.0)}

    def get_solution() -> Solution:
        path : List[Pose] = []
        curr_traceback_pose = end_pose
        path.append(end_pose)

        while curr_traceback_pose != start_pose:
            if curr_traceback_pose not in graph.keys():
                return (None, np.inf)
            prev_pose = graph[curr_traceback_pose].prev
            if prev_pose is None:
                return (None, np.inf)
            path.append(prev_pose)
            curr_traceback_pose = prev_pose
        path.reverse() # path[0] is the start and path[-1] is the end
        return (path, graph[end_pose].cost_from_start)

    # Create a priority queue where the priority is according to the cost from the start
    heap = []
    heapq.heappush(heap, graph[start_pose])
    
    if verbose:
        print(f"Starting Dijkstras!")
    
    # Finishes when the best known cost from the start to each node in the graph is calculated!
    while len(heap) > 0:

        curr_node : Node = heapq.heappop(heap)

        # Get neighbors if we do not have them yet
        if curr_node.neighbors is None:
            curr_node.neighbors = neighbors_cb(curr_node.pose)
        
        if verbose:
            print(f"\tNeighbors of {curr_node.pose}: {curr_node.neighbors}")

        for neighbor_pose in curr_node.neighbors:
            # Calculate cost to neighbor node travelling through curr_node
            cost_through_curr = cost_cb(curr_node.pose, neighbor_pose) + curr_node.cost_from_start
            
            if verbose:
                print(f"\tCost through {curr_node.pose} to {neighbor_pose}: {cost_through_curr}")

            # If neighbor is not included in the graph, add it!
            if neighbor_pose not in graph.keys():
                graph[neighbor_pose] = Node(neighbor_pose)

            # Update the neighbor node's cost if cost_through_curr is better
            neighbor_node = graph[neighbor_pose]
            if cost_through_curr < neighbor_node.cost_from_start:
                
                if verbose:
                    print(f"\t\tFound faster way to reach {neighbor_node.pose}! Updating cost to {cost_through_curr} from {neighbor_node.cost_from_start}.")
                neighbor_node.cost_from_start = cost_through_curr
                neighbor_node.prev = curr_node.pose # Use Bellman's Principle of Optimality to find the best path

                # Add any node that has been updated, i.e., any node that may cause further inconsistencies, to the priority queue
                heapq.heappush(heap, graph[neighbor_pose])

        yield get_solution()