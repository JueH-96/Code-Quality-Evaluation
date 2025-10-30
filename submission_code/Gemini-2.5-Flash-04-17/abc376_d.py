import sys
from collections import deque

def solve():
    # Read input
    # sys.stdin.readline is faster than input() for competitive programming
    N, M = map(int, sys.stdin.readline().split())

    # Build adjacency lists for the original directed graph and its reversed version.
    # The original graph's adjacency list `adj[u]` stores neighbors reachable from `u`.
    # The reversed graph's adjacency list `rev_adj[u]` stores vertices `v` such that there is an edge `v -> u`
    # in the original graph.
    # We also store the edges as a list of tuples to easily iterate through them later.
    # Using 1-based indexing for vertices as per the problem statement, so lists are size N+1.
    adj = [[] for _ in range(N + 1)]
    rev_adj = [[] for _ in range(N + 1)]
    edges = [] # Store edges as (u, v) tuples

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        rev_adj[v].append(u) # Add edge (v, u) to the reversed graph
        edges.append((u, v)) # Store the original edge (u, v)

    # Step 1: BFS from vertex 1 in the original graph.
    # This calculates the shortest distance (minimum number of edges) from vertex 1
    # to every other vertex reachable from 1.
    dist1 = [float('inf')] * (N + 1) # Initialize distances to infinity
    dist1[1] = 0 # Distance from vertex 1 to itself is 0
    q = deque([1]) # Initialize BFS queue with the starting vertex 1

    while q:
        u = q.popleft()
        # Explore neighbors reachable from u in the original graph
        for v in adj[u]:
            # If vertex v has not been visited yet (its distance is still infinity)
            if dist1[v] == float('inf'):
                dist1[v] = dist1[u] + 1 # The distance to v is one more than the distance to u
                q.append(v) # Add v to the queue for further exploration

    # Step 2: BFS from vertex 1 in the reversed graph.
    # This calculates the shortest distance from any vertex `v` to vertex 1 in the original graph.
    # This is equivalent to finding the shortest distance from vertex 1 to `v` in the reversed graph.
    distR = [float('inf')] * (N + 1) # Initialize distances to infinity
    distR[1] = 0 # Distance from vertex 1 to itself in the reversed graph is 0
    q = deque([1]) # Initialize BFS queue with vertex 1 (in the context of the reversed graph)

    while q:
        u = q.popleft()
        # Explore neighbors reachable from u in the reversed graph.
        # These are vertices `v` such that there's an edge `v -> u` in the original graph.
        for v in rev_adj[u]:
            # If vertex v has not been visited yet (its distance is still infinity)
            if distR[v] == float('inf'):
                distR[v] = distR[u] + 1 # The distance to v (in reversed graph) is one more than distance to u
                q.append(v) # Add v to the queue

    # Step 3: Calculate candidate cycle lengths.
    # A cycle containing vertex 1 can be viewed as a path from 1 to some vertex u,
    # followed by an edge (u, v), followed by a path from v back to 1.
    # To find the shortest such cycle, we can iterate through all edges (u, v) in the graph.
    # If there exists a path from 1 to u (length dist1[u]) and a path from v to 1 (length distR[v]),
    # then dist1[u] + 1 (for edge u->v) + distR[v] is the length of a walk from 1 back to 1
    # that includes the edge (u,v). This walk must contain a cycle through 1.
    # The minimum of these lengths over all edges (u, v) is the length of the shortest cycle containing 1.
    min_cycle_len = float('inf') # Initialize minimum cycle length to infinity

    # Iterate through all edges (u, v) in the original graph
    for u, v in edges:
        # For an edge (u, v) to be part of a cycle containing 1,
        # vertex u must be reachable from 1 (dist1[u] is finite) AND
        # vertex v must be able to reach 1 (distR[v] is finite).
        # Vertex 1 is a special case for shortest path calculations (dist1[1]=0, distR[1]=0).
        # The problem states a_i != b_i, so u and v are always different for any edge.
        # The paths must not use the edge (u,v) itself. BFS guarantees this for shortest paths.
        # The cycle formed is 1 --(shortest path)--> u --(edge)--> v --(shortest path)--> 1.
        # The length is dist1[u] + 1 + distR[v].
        if dist1[u] != float('inf') and distR[v] != float('inf'):
             # A simple cycle containing 1 must use a path from 1 to some vertex u,
             # an edge (u, v), and a path from v to 1, such that these paths and the edge form a simple cycle.
             # The shortest such cycle has length min(dist1[u] + 1 + distR[v]) over all edges (u, v).
             candidate_len = dist1[u] + 1 + distR[v]
             # Update the minimum cycle length found so far
             min_cycle_len = min(min_cycle_len, candidate_len)

    # Step 4: Output result
    if min_cycle_len == float('inf'):
        # If min_cycle_len is still infinity after checking all edges,
        # it means no edge (u, v) existed such that u is reachable from 1
        # and v can reach 1. Therefore, no cycle containing vertex 1 exists.
        print(-1)
    else:
        # Otherwise, the minimum cycle length found is the answer.
        print(min_cycle_len)

# Execute the solve function
solve()