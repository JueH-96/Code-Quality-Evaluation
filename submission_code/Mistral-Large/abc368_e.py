import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X_1 = int(data[2])

trains = []
for i in range(M):
    A = int(data[3 + 4 * i])
    B = int(data[4 + 4 * i])
    S = int(data[5 + 4 * i])
    T = int(data[6 + 4 * i])
    trains.append((A, B, S, T))

# Build the graph
graph = [[] for _ in range(N + 1)]
for i, (A, B, S, T) in enumerate(trains):
    graph[B].append((T + X_1 if i == 0 else T, i + 1))

# Dijkstra's algorithm to find the minimum delays
dist = [float('inf')] * (M + 1)
dist[1] = X_1
priority_queue = [(X_1, 1)]

while priority_queue:
    current_delay, current_train = heapq.heappop(priority_queue)
    if current_delay > dist[current_train]:
        continue
    A, B, S, T = trains[current_train - 1]
    for next_time, next_train in graph[B]:
        delay = max(0, S - (T + current_delay))
        if dist[next_train] > delay:
            dist[next_train] = delay
            heapq.heappush(priority_queue, (delay, next_train))

# Output the results
print(' '.join(map(str, dist[2:])))