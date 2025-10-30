import collections
import sys

def solve():
    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, sys.stdin.readline().split())

    # Initialize an adjacency list to represent the graph.
    # adj[u] will store a list of vertices v for which there is a directed edge u -> v.
    # Vertices are 1-indexed, so we create lists for N+1 indices and use indices 1 to N.
    adj = [[] for _ in range(N + 1)]

    # Read M edges and populate the adjacency list
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)

    # dist[i] will store the shortest distance (number of edges) from vertex 1 to vertex i.
    # Initialize with -1 to signify that a vertex has not yet been visited from vertex 1.
    dist = [-1] * (N + 1)

    # Initialize a deque (double-ended queue) for the BFS.
    # The queue will store vertices to be visited.
    q = collections.deque()

    # Start the BFS from vertex 1.
    # The distance from vertex 1 to itself is 0.
    dist[1] = 0
    # Add vertex 1 to the queue to start the traversal.
    q.append(1)

    # Initialize min_cycle_len to a very large value (infinity).
    # This variable will hold the minimum length of a cycle containing vertex 1.
    # If no such cycle is found, it will remain infinity.
    min_cycle_len = float('inf')

    # Perform the BFS traversal
    while q:
        # Dequeue the current vertex to process
        u = q.popleft()

        # Iterate over all neighbors 'v' of the current vertex 'u'
        for v in adj[u]:
            # If the neighbor 'v' is vertex 1, we have found a cycle!
            # This means there's a path from 1 to 'u' (length dist[u])
            # and an edge from 'u' back to 1.
            if v == 1:
                # Calculate the length of this cycle.
                # Since BFS guarantees dist[u] is the shortest path from 1 to u,
                # dist[u] + 1 is a candidate for the shortest cycle length.
                current_cycle_len = dist[u] + 1
                # Update min_cycle_len if this cycle is shorter than any found so far.
                min_cycle_len = min(min_cycle_len, current_cycle_len)
            
            # If the neighbor 'v' has not been visited yet (dist[v] is -1),
            # it means we've found a new shortest path to 'v'.
            # (If dist[v] is not -1, it means 'v' has already been reached by a path
            # that is shorter or equal to the current path, due to BFS properties.)
            if dist[v] == -1:
                # Update the shortest distance to 'v'
                dist[v] = dist[u] + 1
                # Enqueue 'v' to explore its neighbors later
                q.append(v)

    # After the BFS completes, check the value of min_cycle_len.
    if min_cycle_len == float('inf'):
        # If min_cycle_len is still infinity, it means no cycle containing vertex 1 was found.
        print(-1)
    else:
        # Otherwise, print the minimum cycle length found.
        # Cast to int because float('inf') and min_cycle_len operations might result in float.
        print(int(min_cycle_len))

# Call the solve function to execute the program
solve()