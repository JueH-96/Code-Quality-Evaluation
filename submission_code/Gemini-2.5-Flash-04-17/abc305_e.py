# YOUR CODE HERE
import sys
from collections import deque

# Increase recursion depth if necessary (though BFS is iterative)
# sys.setrecursionlimit(200500)

def solve():
    # Read N, M, K: number of vertices, edges, and guards
    N, M, K = map(int, sys.stdin.readline().split())

    # Build adjacency list for the graph (1-indexed vertices)
    # adj[i] will contain a list of vertices adjacent to vertex i
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Add edge (u, v) for undirected graph
        adj[u].append(v)
        adj[v].append(u)

    # max_reach[v] stores the maximum 'effective stamina' that reaches vertex v
    # from any guard. Effective stamina = initial_stamina - distance.
    # Initialize with -1 to indicate no effective reach yet.
    max_reach = [-1] * (N + 1)

    # Queue for the multi-source BFS
    # Stores vertices that need to be processed
    q = deque()

    # Initialize the BFS with the guards' positions and stamina
    for _ in range(K):
        p, h = map(int, sys.stdin.readline().split())
        # Guard i is at vertex p_i with stamina h_i.
        # This means vertex p_i is "reached" with an initial effective stamina of h_i.
        # We take the maximum stamina if multiple guards were at the same location,
        # although the problem states p_i are distinct.
        if h > max_reach[p]: # This check is effectively always true with initial -1
             max_reach[p] = h
             # Add the guard's location to the queue to start the BFS propagation
             q.append(p)

    # Perform the multi-source BFS to propagate the maximum reach
    while q:
        # Get the vertex with the current maximum reach from the queue
        u = q.popleft()
        r = max_reach[u] # Current maximum reach at vertex u

        # If the current reach is 0 or less, it means the "stamina" is exhausted
        # or negative, and cannot positively contribute to guarding further vertices
        # (as propagating would result in a reach <= -1).
        if r <= 0:
             continue

        # Explore neighbors of u
        for v in adj[u]:
            # The effective reach decreases by 1 for each edge traversed
            nr = r - 1 # New potential reach for neighbor v through u

            # If this new reach (nr) is greater than the current best reach recorded for v
            if nr > max_reach[v]:
                # Update max_reach[v] and add v to the queue
                # A vertex is added to the queue if a better reach is found for it,
                # ensuring the BFS explores paths that yield higher remaining stamina.
                max_reach[v] = nr
                q.append(v)

    # Collect all vertices v where max_reach[v] >= 0
    # A vertex v is guarded if there exists at least one guard i such that
    # dist(v, p_i) <= h_i. This is equivalent to max_i(h_i - dist(v, p_i)) >= 0,
    # which is exactly what max_reach[v] >= 0 signifies.
    # The list comprehension naturally collects vertices in ascending order (1 to N).
    guarded_vertices = [v for v in range(1, N + 1) if max_reach[v] >= 0]

    # Print the number of guarded vertices
    print(len(guarded_vertices))

    # Print the guarded vertices in ascending order, space-separated
    print(*guarded_vertices)

# Execute the solve function
solve()