import sys
import heapq

def dijkstra(N, D_mat, mult, add, start):
    dist = [float('inf') for _ in range(N)]
    dist[start] = 0
    pq = []
    heapq.heappush(pq, (0, start))
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > dist[u]:
            continue
        for v in range(N):
            if u == v:
                continue
            weight = D_mat[u][v] * mult + add
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    return dist

data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
A = data[index + 1]
B = data[index + 2]
C = data[index + 3]
index += 4

D_matrix = []
for i in range(N):
    row = data[index:index + N]
    index += N
    D_matrix.append(row)

dist_car_from_start = dijkstra(N, D_matrix, A, 0, 0)
dist_train_from_end = dijkstra(N, D_matrix, B, C, N - 1)

min_time = min(dist_car_from_start[k] + dist_train_from_end[k] for k in range(N))
print(int(min_time))