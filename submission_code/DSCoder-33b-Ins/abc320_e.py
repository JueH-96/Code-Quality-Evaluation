# YOUR CODE HERE
import heapq

N, M = map(int, input().split())

events = []
for _ in range(M):
    T, W, S = map(int, input().split())
    events.append((T, W, S))

events.sort()

queue = []
noodles = [0] * (N + 1)
i = 0

for t in range(1, N + 1):
    while i < M and events[i][0] <= t:
        heapq.heappush(queue, events[i][1:])
        i += 1
    while queue and queue[0][1] < t:
        heapq.heappop(queue)
    if queue:
        noodles[t] = queue[0][0]

print(*noodles[1:], sep='
')