import heapq

n, A, B, C = map(int, input().split())
D = [list(map(int, input().split())) for _ in range(n)]

INF = float('inf')
dist = [[INF] * 2 for _ in range(n)]
dist[0][0] = 0
dist[0][1] = 0

heap = []
heapq.heappush(heap, (0, 0, 0))
heapq.heappush(heap, (0, 0, 1))

while heap:
    time, u, mode = heapq.heappop(heap)
    if time > dist[u][mode]:
        continue
    if u == n - 1:
        continue
    if mode == 0:
        for v in range(n):
            if v == u:
                continue
            new_time = time + D[u][v] * A
            if new_time < dist[v][0]:
                dist[v][0] = new_time
                heapq.heappush(heap, (new_time, v, 0))
        if time < dist[u][1]:
            dist[u][1] = time
            heapq.heappush(heap, (time, u, 1))
    else:
        for v in range(n):
            if v == u:
                continue
            new_time = time + D[u][v] * B + C
            if new_time < dist[v][1]:
                dist[v][1] = new_time
                heapq.heappush(heap, (new_time, v, 1))

print(min(dist[n-1][0], dist[n-1][1]))