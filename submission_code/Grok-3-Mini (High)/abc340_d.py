import heapq
import sys

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Read A, B, X lists
A_list = []
B_list = []
X_list = []
for _ in range(N - 1):
    A_val = int(data[index])
    index += 1
    B_val = int(data[index])
    index += 1
    X_val = int(data[index])
    index += 1
    A_list.append(A_val)
    B_list.append(B_val)
    X_list.append(X_val)

# Dijkstra's algorithm
INF = 10**15
dist = [INF] * (N + 1)
dist[1] = 0
heap = []
heapq.heappush(heap, (0, 1))  # (distance, node)

while heap:
    cur_dist, u = heapq.heappop(heap)
    if cur_dist > dist[u]:
        continue  # Skip outdated entries
    if u < N:
        # Relax edge to u+1
        nei = u + 1
        weight_a = A_list[u - 1]
        if dist[nei] > cur_dist + weight_a:
            dist[nei] = cur_dist + weight_a
            heapq.heappush(heap, (dist[nei], nei))
        
        # Relax edge to X_u
        x_nei = X_list[u - 1]
        weight_b = B_list[u - 1]
        if dist[x_nei] > cur_dist + weight_b:
            dist[x_nei] = cur_dist + weight_b
            heapq.heappush(heap, (dist[x_nei], x_nei))

# Output the shortest distance to node N
print(dist[N])