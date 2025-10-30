import heapq
import sys

# Read input
N, M, X = map(int, sys.stdin.readline().split())

# Create adjacency list for the 2N-node graph
# Nodes 1 to N represent vertices 1 to N in the original graph state (state 0)
# Nodes N+1 to 2N represent vertices 1 to N in the reversed graph state (state 1)
# Node v (1 <= v <= N) corresponds to vertex v in state 0
# Node v + N (1 <= v <= N) corresponds to vertex v in state 1
# Total nodes will be 2*N. We use 1-based indexing for nodes, so size 2*N + 1
adj = [[] for _ in range(2 * N + 1)]

# Add edges from the original graph and reversed graph
# For an edge u -> v in the input graph G:
# 1. Movement in G: (u, 0) -> (v, 0) with cost 1. In 2N graph: u -> v cost 1.
# 2. Movement in G_rev: (v, 1) -> (u, 1) with cost 1 (since u -> v in G means v -> u in G_rev).
#    In 2N graph: v + N -> u + N cost 1.
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    # Edge u -> v in original graph G
    adj[u].append((v, 1))
    # Edge v -> u in reversed graph G_rev
    adj[v + N].append((u + N, 1))

# Add edges for the reversing operation
# For each vertex i from 1 to N:
# 1. Reverse from state 0 to state 1 at vertex i: (i, 0) -> (i, 1) with cost X.
#    In 2N graph: i -> i + N cost X.
# 2. Reverse from state 1 to state 0 at vertex i: (i, 1) -> (i, 0) with cost X.
#    In 2N graph: i + N -> i cost X.
for v in range(1, N + 1):
    adj[v].append((v + N, X))
    adj[v + N].append((v, X))

# Dijkstra's algorithm
# Distance array initialized to infinity. Use a value larger than any possible path cost.
# Python's float('inf') works for comparison, and integer addition handles large costs.
dist = [float('inf')] * (2 * N + 1)

# Start at vertex 1 in original graph state (node 1)
start_node = 1
dist[start_node] = 0

# Priority queue storing tuples (cost, node). heapq is a min-heap.
pq = [(0, start_node)]

while pq:
    # Pop the node with the smallest distance
    current_cost, current_node = heapq.heappop(pq)

    # If we have found a shorter path already, skip processing this one
    if current_cost > dist[current_node]:
        continue

    # Explore neighbors
    for neighbor_node, edge_weight in adj[current_node]:
        new_cost = current_cost + edge_weight

        # If the new path to the neighbor is shorter
        if new_cost < dist[neighbor_node]:
            dist[neighbor_node] = new_cost
            # Push the neighbor with the new cost into the priority queue
            heapq.heappush(pq, (new_cost, neighbor_node))

# The target is vertex N. It can be reached in either state 0 (node N) or state 1 (node 2N).
target_node_state0 = N
target_node_state1 = 2 * N

# The minimum cost is the minimum of the shortest distances to the target nodes
min_cost = min(dist[target_node_state0], dist[target_node_state1])

# Print the result
print(min_cost)