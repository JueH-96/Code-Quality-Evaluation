import heapq
import sys

def solve():
    # Read N, M, X
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    X = int(line1[2])

    # Adjacency list for the 2N state graph
    # Vertices 1 to N: state (v, 0) - original orientation
    # Vertices N+1 to 2N: state (v, 1) - reversed orientation (vertex v in reversed graph)
    # Vertex v corresponds to state (v, 0)
    # Vertex v + N corresponds to state (v, 1)
    adj = [[] for _ in range(2 * N + 1)]

    # Add edges for movement along directed edges
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Edge u -> v in original graph (cost 1)
        # Represents movement from state (u, 0) to (v, 0)
        adj[u].append((v, 1))
        # Edge v -> u in reversed graph (cost 1)
        # Represents movement from state (v, 1) to (u, 1)
        # Vertex v in reversed graph is state (v, 1), represented by v + N
        # Vertex u in reversed graph is state (u, 1), represented by u + N
        adj[v + N].append((u + N, 1))

    # Add edges for reversing orientation (cost X)
    # From state (v, 0) to (v, 1): vertex v to vertex v + N with cost X
    # From state (v, 1) to (v, 0): vertex v + N to vertex v with cost X
    for i in range(1, N + 1):
        adj[i].append((i + N, X))
        adj[i + N].append((i, X))

    # Dijkstra's algorithm
    # dist array to store minimum cost to reach each state
    dist = [float('inf')] * (2 * N + 1)
    
    # Start state is vertex 1 in original orientation (state (1, 0))
    start_node = 1 
    dist[start_node] = 0

    # Priority queue stores (cost, vertex)
    pq = [(0, start_node)]

    while pq:
        current_cost, u = heapq.heappop(pq)

        # If we found a shorter path already, skip
        if current_cost > dist[u]:
            continue

        # Explore neighbors
        for v, weight in adj[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    # The target is vertex N in either orientation
    # State (N, 0) is vertex N
    # State (N, 1) is vertex N + N = 2N
    min_cost = min(dist[N], dist[2 * N])

    # The problem guarantees reachability, so min_cost should not be infinity.
    sys.stdout.write(str(min_cost) + '
')

# Execute the solver function
solve()