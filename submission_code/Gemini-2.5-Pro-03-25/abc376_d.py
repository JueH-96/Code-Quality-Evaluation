# YOUR CODE HERE
import sys
import collections

def solve():
    """
    Solves the problem of finding the minimum length cycle containing vertex 1
    in a directed graph. Uses Breadth-First Search (BFS) to find shortest paths
    from vertex 1, then checks paths ending with an edge back to vertex 1.
    """
    
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency list representation of the graph.
    # adj[u] contains a list of vertices v such that there is an edge u -> v.
    adj = collections.defaultdict(list)
    
    # List to store vertices that have a direct edge pointing to vertex 1.
    # These are potential predecessors of vertex 1 in a cycle.
    predecessors_of_1 = []
    
    # Read M edges and build the adjacency list and list of predecessors of 1.
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        
        # If an edge points to vertex 1, record its source vertex u.
        if v == 1:
            predecessors_of_1.append(u)

    # Initialize BFS queue. Start BFS from vertex 1.
    # The queue stores tuples: (vertex, distance_from_start_node_1)
    q = collections.deque([(1, 0)]) 
    
    # Dictionary to store the shortest distance found so far from vertex 1 to any vertex `v`.
    # Key: vertex, Value: shortest distance (number of edges).
    # Initialize distance for the starting vertex 1 as 0.
    dist = {1: 0} 

    # Perform Breadth-First Search starting from vertex 1.
    while q:
        # Get the current vertex and its distance from vertex 1.
        curr_v, curr_dist = q.popleft()

        # Explore neighbors of the current vertex.
        for neighbor in adj[curr_v]:
            # Check if the neighbor has been visited yet (i.e., its shortest distance from 1 is not yet determined).
            # Using `neighbor not in dist` checks if the key `neighbor` exists in the dictionary `dist`.
            if neighbor not in dist:
                 # If not visited, record its distance = current distance + 1.
                 dist[neighbor] = curr_dist + 1
                 # Add the neighbor to the queue for further exploration.
                 q.append((neighbor, curr_dist + 1))
            # If the neighbor is already in dist, it means we have found a path to it before.
            # Standard BFS guarantees that the first time we reach a node, it is via a shortest path.
            # Therefore, we do not need to update the distance or enqueue it again.
            
    # After BFS completion, `dist` contains shortest path distances from vertex 1 to all reachable vertices.
    
    # Initialize minimum cycle length to infinity.
    min_cycle_len = float('inf')

    # Check all vertices `p_v` that have a direct edge to vertex 1.
    for p_v in predecessors_of_1:
        # Check if this predecessor vertex `p_v` is reachable from vertex 1.
        # Reachability is confirmed if `p_v` is a key in the `dist` dictionary.
        if p_v in dist:
            # If `p_v` is reachable, we found a cycle: 1 -> ... -> p_v -> 1.
            # The path 1 -> ... -> p_v has length `dist[p_v]`.
            # The edge p_v -> 1 adds 1 to the length.
            cycle_len = dist[p_v] + 1
            # Update the minimum cycle length found so far.
            min_cycle_len = min(min_cycle_len, cycle_len)

    # Check if any cycle containing vertex 1 was found.
    if min_cycle_len == float('inf'):
        # If min_cycle_len is still infinity, no such cycle exists.
        print("-1")
    else:
        # Otherwise, print the minimum cycle length found.
        print(min_cycle_len)

# Ensure the solve function is called when the script is executed.
if __name__ == '__main__':
    solve()