# YOUR CODE HERE
from collections import defaultdict
from heapq import heappush, heappop

N, M = map(int, input().split())
trains = defaultdict(list)

for _ in range(M):
    l, d, k, c, A, B = map(int, input().split())
    for i in range(k):
        trains[A].append((l + i * d, B, c))

def dijkstra(start):
    dist = [-float('inf')] * (N + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        time, station = heappop(pq)
        if time > dist[station]:
            continue
        for next_time, next_station, c in trains[station]:
            if next_time + c > time:
                continue
            if dist[next_station] < next_time + c:
                dist[next_station] = next_time + c
                heappush(pq, (next_time + c, next_station))
    return dist

distances = dijkstra(1)
for i in range(2, N):
    if distances[i] == -float('inf'):
        print("Unreachable")
    else:
        print(distances[i])