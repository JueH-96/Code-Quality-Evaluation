import heapq

n, A, B, C = map(int, input().split())
D = [list(map(int, input().split())) for _ in range(n)]

INF = float('inf')
dist = [[INF] * 2 for _ in range(n)]
dist[0][0] = 0

heap = []
heapq.heappush(heap, (0, 0, 0))

while heap:
    current_dist, i, mode = heapq.heappop(heap)
    if current_dist > dist[i][mode]:
        continue
    for j in range(n):
        if j == i:
            continue
        if mode == 0:
            cost_car = D[i][j] * A
            new_dist = current_dist + cost_car
            if new_dist < dist[j][0]:
                dist[j][0] = new_dist
                heapq.heappush(heap, (new_dist, j, 0))
            cost_train = D[i][j] * B + C
            new_dist = current_dist + cost_train
            if new_dist < dist[j][1]:
                dist[j][1] = new_dist
                heapq.heappush(heap, (new_dist, j, 1))
        else:
            cost_train = D[i][j] * B + C
            new_dist = current_dist + cost_train
            if new_dist < dist[j][1]:
                dist[j][1] = new_dist
                heapq.heappush(heap, (new_dist, j, 1))

ans = min(dist[n-1][0], dist[n-1][1])
print(ans)