# YOUR CODE HERE
import sys
from heapq import heappop, heappush

input = sys.stdin.readline

N, A, B, C = map(int, input().split())
D = [list(map(int, input().split())) for _ in range(N)]

pq = [(0, 0, 0)]  # (time, city, last_transport)
visited = [[False] * N for _ in range(N)]

while pq:
    time, city, last_transport = heappop(pq)
    if city == N - 1:
        print(time)
        break
    if visited[city][last_transport]:
        continue
    visited[city][last_transport] = True
    for next_city in range(N):
        if D[city][next_city] == 0:
            continue
        if last_transport == 0:  # company car
            heappush(pq, (time + D[city][next_city] * A, next_city, 0))
            if next_city != N - 1:
                heappush(pq, (time + D[city][next_city] * A + B + C, next_city, 1))
        else:  # train
            heappush(pq, (time + B + C + D[city][next_city] * B, next_city, 1))