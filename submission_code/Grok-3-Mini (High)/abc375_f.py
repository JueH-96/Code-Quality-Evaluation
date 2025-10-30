import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Initialize adjacency matrix
adj = [[float('inf') for _ in range(N)] for _ in range(N)]
for i in range(N):
    adj[i][i] = 0

# Store road data for closing roads
road_data = []
for _ in range(M):
    A = int(data[index]) - 1  # Convert to 0-based
    index += 1
    B = int(data[index]) - 1  # Convert to 0-based
    index += 1
    C = int(data[index])
    index += 1
    adj[A][B] = C
    adj[B][A] = C
    road_data.append((A, B))

# Function to compute all-pairs shortest paths using Floyd-Warshall
def compute_dist(adj_mat, N):
    dist = [row[:] for row in adj_mat]  # Deep copy of adjacency matrix
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# Initial computation of shortest paths
dist = compute_dist(adj, N)

# Process Q queries
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        # Close road i
        i = int(data[index])  # Road index is 1-based
        index += 1
        a, b = road_data[i - 1]  # Get the cities connected by the road
        adj[a][b] = float('inf')
        adj[b][a] = float('inf')
        # Recompute shortest paths
        dist = compute_dist(adj, N)
    elif query_type == 2:
        # Query shortest distance between x and y
        x = int(data[index])  # City x is 1-based
        index += 1
        y = int(data[index])  # City y is 1-based
        index += 1
        dist_val = dist[x - 1][y - 1]
        if dist_val < float('inf'):
            print(int(dist_val))
        else:
            print(-1)