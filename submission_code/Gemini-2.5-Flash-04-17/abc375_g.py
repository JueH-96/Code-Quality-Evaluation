import heapq
import sys

def dijkstra(start, N, graph):
    dist = [float('inf')] * (N + 1)
    dist[start] = 0
    pq = [(0, start)] # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        # Check if dist[u] is finite before iterating neighbors to avoid float('inf') + w
        if dist[u] == float('inf'):
             continue

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Store original edges with their index
    original_edges = []
    # Adjacency list for the original graph
    graph = [[] for _ in range(N + 1)]

    for i in range(M):
        u, v, c = map(int, sys.stdin.readline().split())
        original_edges.append((u, v, c, i))
        graph[u].append((v, c))
        graph[v].append((u, c))

    # Run Dijkstra from city 1
    dist1 = dijkstra(1, N, graph)

    # Run Dijkstra from city N (can be done on the same graph since roads are bidirectional)
    distN = dijkstra(N, N, graph)

    # Shortest distance from 1 to N in the original graph
    dist_all = dist1[N] # Guaranteed finite by problem constraints

    # Build the shortest path DAG (G'_sp)
    # A directed edge u -> v exists in G'_sp if the original edge (u, v, w) is part of
    # *some* shortest path from 1 to N. This condition is dist1[u] + w + distN[v] == dist_all.
    # We need adjacency lists for G'_sp and its reverse for path counting.
    G_sp_adj = [[] for _ in range(N + 1)]
    G_sp_rev_adj = [[] for _ in range(N + 1)]

    for u0, v0, w0, i in original_edges:
        # Check if edge (u0, v0, w0) is on a shortest path from 1 to N.
        # This can happen in two "directions" corresponding to directed edges in G'_sp.
        
        # Check u0 -> v0 direction for G'_sp
        # dist1[u0] and distN[v0] must be finite if dist1[u0] + w0 + distN[v0] == dist_all (finite)
        if dist1[u0] != float('inf') and distN[v0] != float('inf') and dist1[u0] + w0 + distN[v0] == dist_all:
            G_sp_adj[u0].append((v0, i))
            G_sp_rev_adj[v0].append((u0, i))
        # Check v0 -> u0 direction for G'_sp
        elif dist1[v0] != float('inf') and distN[u0] != float('inf') and dist1[v0] + w0 + distN[u0] == dist_all:
             G_sp_adj[v0].append((u0, i))
             G_sp_rev_adj[u0].append((v0, i))

    # Compute the number of shortest paths from 1 to each node v in G'_sp (count[v])
    # This is done using dynamic programming on the DAG G'_sp.
    # The vertices can be processed in topological order, which is consistent with
    # increasing dist1 values for edges in G'_sp.
    count = [0] * (N + 1)
    count[1] = 1 # Base case: there is 1 path from 1 to 1 (the empty path)

    # Sort vertices based on their shortest distance from 1
    sorted_vertices = sorted(range(1, N + 1), key=lambda v: dist1[v])

    for u in sorted_vertices:
        # If u is not reachable from 1 via a shortest path (dist1[u] is inf or count[u] is 0),
        # it cannot contribute to path counts of its neighbors in G'_sp.
        if dist1[u] == float('inf') or count[u] == 0:
             continue

        # Propagate count from u to its neighbors v in G'_sp
        for v, original_idx in G_sp_adj[u]:
            count[v] += count[u]

    # Compute the number of shortest paths from each node v to N in G'_sp (rev_count[v])
    # This is also done using dynamic programming on the DAG G'_sp.
    # The vertices can be processed in reverse topological order, which is consistent with
    # decreasing dist1 values.
    rev_count = [0] * (N + 1)
    rev_count[N] = 1 # Base case: there is 1 path from N to N (the empty path)

    # Sort vertices in decreasing order of their shortest distance from 1
    # This gives a reverse topological order for G'_sp
    sorted_vertices_desc = sorted(range(1, N + 1), key=lambda v: dist1[v], reverse=True)

    for v in sorted_vertices_desc:
        # If v cannot reach N via a shortest path in G'_sp (rev_count[v] is 0),
        # it cannot contribute to path counts of its predecessors in G'_sp.
        # Note: If dist1[v] == float('inf'), rev_count[v] will be 0 unless v is N.
        # The DP ensures this implicitly.
        if rev_count[v] == 0:
            continue

        # Propagate rev_count from v to its predecessors u in G'_sp (edges u -> v)
        # These are the neighbors of v in the reverse graph G_sp_rev_adj
        for u, original_idx in G_sp_rev_adj[v]:
            rev_count[u] += rev_count[v]

    # The total number of shortest paths from 1 to N in G'_sp
    total_paths = count[N]

    # Initialize results array
    results = ["No"] * M

    # For each original edge, determine if removing it changes the shortest distance
    for u0, v0, w0, i in original_edges:
        is_essential = False
        
        # If the edge (u0, v0, w0) is on a shortest path, it corresponds to a directed edge in G'_sp.
        # There is at most one such directed edge per original edge due to positive weights.
        
        # Case 1: u0 -> v0 is the directed edge in G'_sp
        if dist1[u0] != float('inf') and distN[v0] != float('inf') and dist1[u0] + w0 + distN[v0] == dist_all:
            # The edge u0 -> v0 is in G'_sp. Its removal changes the shortest distance
            # if and only if it is part of *every* shortest path from 1 to N.
            # This is true iff (number of paths 1->u0 in G'_sp) * (number of paths v0->N in G'_sp)
            # equals the total number of paths 1->N in G'_sp.
            if count[u0] * rev_count[v0] == total_paths:
                is_essential = True
        # Case 2: v0 -> u0 is the directed edge in G'_sp
        elif dist1[v0] != float('inf') and distN[u0] != float('inf') and dist1[v0] + w0 + distN[u0] == dist_all:
             # The edge v0 -> u0 is in G'_sp. Check if it's essential as above.
             if count[v0] * rev_count[u0] == total_paths:
                 is_essential = True
        # Case 3: The original edge is not on any shortest path (neither direction forms an edge in G'_sp).
        # Removing such an edge does not remove any shortest path of length dist_all.
        # Thus, the shortest distance remains dist_all (assuming N is still reachable).
        # The problem constraints and samples suggest that if an edge is not on a shortest path,
        # its removal does not change the shortest distance or make N unreachable in a way
        # that isn't covered by the "essential shortest path edge" condition.
        # The default answer for non-shortest path edges is "No".

        if is_essential:
            results[i] = "Yes"

    # Print the results for each original edge
    for res in results:
        print(res)

solve()