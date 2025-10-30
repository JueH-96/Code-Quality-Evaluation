import sys
import heapq

# Read all input data
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read vertex weights A, make it 1-based
A = [0]  # A[0] is unused
for _ in range(N):
    A.append(int(data[index]))
    index += 1

# Build adjacency list with modified edge weights
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    U = int(data[index])
    V = int(data[index + 1])
    B = int(data[index + 2])
    index += 3
    # Add edges with weights B + A[neighbor]
    adj[U].append((V, B + A[V]))
    adj[V].append((U, B + A[U]))

# Dijkstra's algorithm
dist = [float('inf')] * (N + 1)
dist[1] = A[1]  # Start distance includes A[1]
pq = []
heapq.heappush(pq, (dist[1], 1))  # (distance, node)

while pq:
    curr_dist, node = heapq.heappop(pq)
    if curr_dist > dist[node]:
        continue  # Skip outdated entries
    for neigh, weight in adj[node]:
        new_dist = curr_dist + weight
        if new_dist < dist[neigh]:
            dist[neigh] = new_dist
            heapq.heappush(pq, (new_dist, neigh))

# Output distances from node 2 to N
output_str = ' '.join(str(dist[i]) for i in range(2, N + 1))
print(output_str)