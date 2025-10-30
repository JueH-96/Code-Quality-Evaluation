# YOUR CODE HERE
import sys
import heapq

# Read N, M, X
# N: number of vertices
# M: number of edges
# X: cost to reverse edges
N, M, X = map(int, sys.stdin.readline().split())

# Adjacency lists for the original graph G and the reversed graph G'
# adj[u] stores vertices v such that there is a directed edge u -> v in G
# rev_adj[v] stores vertices u such that there is a directed edge u -> v in G' (i.e., u -> v in G)
adj = [[] for _ in range(N + 1)]
rev_adj = [[] for _ in range(N + 1)]

# Read edges and build adjacency lists
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    adj[u].append(v)
    rev_adj[v].append(u) # Edge v -> u in G' corresponds to u -> v in G

# Distance array dist[vertex][orientation]
# dist[v][0]: minimum cost to reach vertex v in the original graph orientation
# dist[v][1]: minimum cost to reach vertex v in the reversed graph orientation
# Initialize with infinity
dist = [[float('inf')] * 2 for _ in range(N + 1)]

# Starting state: vertex 1, original graph (orientation 0), cost 0
dist[1][0] = 0

# Priority queue for Dijkstra's algorithm
# Stores tuples: (current_cost, vertex, orientation)
# The queue is ordered by current_cost (smallest first)
pq = [(0, 1, 0)]

# Dijkstra's algorithm
while pq:
    # Get the state with the minimum cost
    current_cost, u, current_orient = heapq.heappop(pq)

    # If we already found a cheaper path, skip
    if current_cost > dist[u][current_orient]:
        continue

    # If we reached the target vertex N, print the cost and exit
    # Due to the nature of Dijkstra's and non-negative edge weights,
    # the first time we extract a state corresponding to vertex N
    # (in either orientation), its cost is the minimum cost to reach N overall.
    if u == N:
        print(current_cost)
        sys.exit()

    # --- Explore possible next states ---

    # Option 1: Move along a directed edge (cost 1)
    if current_orient == 0: # Currently in the original graph G
        # Iterate through neighbors reachable from u in G
        for v in adj[u]:
            move_cost = current_cost + 1
            # If this path to (v, 0) is cheaper than the current best
            if move_cost < dist[v][0]:
                dist[v][0] = move_cost
                # Add the new state to the priority queue
                heapq.heappush(pq, (move_cost, v, 0))
    else: # Currently in the reversed graph G'
        # Iterate through neighbors reachable from u in G' (i.e., u such that u -> v in G)
        for v in rev_adj[u]:
            move_cost = current_cost + 1
            # If this path to (v, 1) is cheaper than the current best
            if move_cost < dist[v][1]:
                dist[v][1] = move_cost
                # Add the new state to the priority queue
                heapq.heappush(pq, (move_cost, v, 1))

    # Option 2: Reverse edges (cost X)
    # From state (u, current_orient) to (u, 1 - current_orient)
    next_orient = 1 - current_orient
    reverse_cost = current_cost + X
    if reverse_cost < dist[u][next_orient]:
        dist[u][next_orient] = reverse_cost
        # Add the new state to the priority queue
        heapq.heappush(pq, (reverse_cost, u, next_orient))

# The problem guarantees reachability to N, so sys.exit() should be hit before the loop finishes.
# If it wasn't guaranteed, we would print min(dist[N][0], dist[N][1]) here,
# potentially printing infinity if N is unreachable.