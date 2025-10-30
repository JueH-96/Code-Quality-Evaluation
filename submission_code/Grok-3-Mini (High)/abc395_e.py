import sys
import heapq

# Read all input data
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
M = data[index + 1]
X = data[index + 2]
index += 3

# Build adjacency lists for out-neighbors and in-neighbors in normal graph
out_adj = [[] for _ in range(N + 1)]
in_adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u = data[index]
    v = data[index + 1]
    out_adj[u].append(v)
    in_adj[v].append(u)  # Edge u -> v, so u is in-neighbor of v
    index += 2

# Dijkstra's algorithm
dist = [[float('inf'), float('inf')] for _ in range(N + 1)]  # dist[v][orient]
dist[1][0] = 0  # Start at vertex 1, orientation 0
pq = []
heapq.heappush(pq, (0, 1, 0))  # (distance, vertex, orientation)

while pq:
    curr_dist, curr_v, curr_orient = heapq.heappop(pq)
    
    # Skip if this is an outdated entry
    if curr_dist > dist[curr_v][curr_orient]:
        continue
    
    # Reverse action: stay at same vertex, flip orientation, cost X
    new_orient_rev = 1 - curr_orient
    new_dist_rev = curr_dist + X
    if new_dist_rev < dist[curr_v][new_orient_rev]:
        dist[curr_v][new_orient_rev] = new_dist_rev
        heapq.heappush(pq, (new_dist_rev, curr_v, new_orient_rev))
    
    # Move action: depend on current orientation
    if curr_orient == 0:
        # In normal orientation, move to out-neighbors
        for nei in out_adj[curr_v]:
            new_v = nei
            new_orient = 0  # Orientation does not change
            new_dist = curr_dist + 1
            if new_dist < dist[new_v][new_orient]:
                dist[new_v][new_orient] = new_dist
                heapq.heappush(pq, (new_dist, new_v, new_orient))
    else:  # curr_orient == 1, reversed orientation
        # In reversed orientation, move to in-neighbors of normal graph
        for nei in in_adj[curr_v]:
            new_v = nei
            new_orient = 1  # Orientation does not change
            new_dist = curr_dist + 1
            if new_dist < dist[new_v][new_orient]:
                dist[new_v][new_orient] = new_dist
                heapq.heappush(pq, (new_dist, new_v, new_orient))

# The minimum cost to reach vertex N in any orientation
min_cost = min(dist[N][0], dist[N][1])
print(min_cost)