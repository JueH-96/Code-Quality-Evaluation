from typing import List

def find_directed_cycle(N: int, A: List[int]) -> List[int]:
    """
    Finds a directed cycle in a graph with N vertices and N edges.
    Each edge goes from vertex i to vertex A[i].
    Returns a list of vertices that form a directed cycle.
    """
    # Adjusting indices to be 0-based
    A = [a - 1 for a in A]

    # Attempting to find a cycle starting from each vertex
    for start_vertex in range(N):
        path = [start_vertex]
        current_vertex = A[start_vertex]

        # Traversing the graph until a cycle is found or we reach a dead end
        while current_vertex not in path:
            path.append(current_vertex)
            current_vertex = A[current_vertex]

        # If a cycle is found, return the cycle
        if current_vertex in path:
            cycle_start = path.index(current_vertex)
            return [vertex + 1 for vertex in path[cycle_start:]]  # Adjusting indices back to 1-based

    # If no cycle is found, this line should not be reached due to problem constraints
    raise Exception("No directed cycle found, which should not happen under the given constraints.")

# Reading input
N = int(input())
A = list(map(int, input().split()))

# Finding and printing the directed cycle
cycle = find_directed_cycle(N, A)
print(len(cycle))
print(' '.join(map(str, cycle)))