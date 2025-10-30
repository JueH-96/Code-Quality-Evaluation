from heapq import heappush, heappop

N, M = map(int, input().split())
events = [tuple(map(int, input().split())) for _ in range(M)]
events.append((10**10, 0, 0))
events.sort(key=lambda x: x[0])
answer = [0] * N
queue = []
time = 0
idx = 0
for i in range(1, N + 1):
    while idx < M and events[idx][0] < time + i:
        T, W, S = events[idx]
        heappush(queue, (T + S, W))
        idx += 1
    if queue:
        T, W = heappop(queue)
        answer[i - 1] += W
        if T < time + N:
            heappush(queue, (T, W))
    time += 1
print('
'.join(map(str, answer)))