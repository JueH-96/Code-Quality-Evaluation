import sys
import heapq

N, M = map(int, input().split())
events = [list(map(int, input().split())) for _ in range(M)]
events.sort()

queue = []
noodles = [0]*N
for i in range(N):
    heapq.heappush(queue, (0, i))

for T, W, S in events:
    while queue and queue[0][0] < T:
        _, i = heapq.heappop(queue)
        heapq.heappush(queue, (T+S, i))
    if queue and queue[0][0] == T:
        _, i = heapq.heappop(queue)
        noodles[i] += W
        heapq.heappush(queue, (T+S, i))

for noodle in noodles:
    print(noodle)