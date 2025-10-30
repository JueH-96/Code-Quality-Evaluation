import sys
from collections import deque

def solve():
    """
    Reads a directed graph, finds the minimum length of a cycle containing vertex 1,
    and prints the result.
    """
    try:
        # Read the number of vertices and edges
        N, M = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle empty input
        return

    # Adjacency list representation of the graph
    adj = [[] for _ in range(N + 1)]
    # List to store all vertices that have an edge pointing to vertex 1
    predecessors_of_1 = []

    # Read all edges and build the graph
    for _ in range(M):
        try:
            u, v = map(int, sys.stdin.readline().split())
            adj[u].append(v)
            if v == 1:
                predecessors_of_1.append(u)
        except (IOError, ValueError):
            continue

    min_cycle_len = float('inf')

    # A cycle through vertex 1 is of the form: 1 -> ... -> p -> 1.
    # The length is shortest_path_length(1, p) + 1.
    # We can find the shortest path from 1 to all its predecessors by running
    # one BFS from each predecessor on the reversed graph, but that's slow.
    #
    # A better approach is to run one BFS from vertex 1 on the forward graph.
    # This gives us shortest_path_length(1, p) for all reachable predecessors p.

    # We run a separate BFS starting from each of 1's neighbors to find the
    # shortest path back to 1.
    for start_node in adj[1]:
        # For each path 1 -> start_node, find the shortest path from start_node back to 1
        q = deque([(start_node, 1)])  # (current_node, path_length_from_start_node)
        # Use a dictionary for visited set for potentially better performance with sparse graphs
        visited = {start_node}
        
        path_found = False
        while q:
            curr_node, length = q.popleft()

            # If the current path is already longer than the best cycle found, prune search
            if length + 1 >= min_cycle_len:
                continue

            for neighbor in adj[curr_node]:
                if neighbor == 1:
                    # Found a path back to 1, completing a cycle
                    min_cycle_len = min(min_cycle_len, length + 1)
                    path_found = True
                    # Since BFS finds the shortest path, we can break this inner loop
                    # and move to the next neighbor of 1.
                    break
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, length + 1))
            
            if path_found:
                break

    # Output the result
    if min_cycle_len == float('inf'):
        print(-1)
    else:
        print(min_cycle_len)

if __name__ == "__main__":
    solve()