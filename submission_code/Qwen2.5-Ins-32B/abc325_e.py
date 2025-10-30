import sys
import heapq

def dijkstra(N, A, B, C, D):
    INF = float('inf')
    dist = [[INF] * 2 for _ in range(N)]
    dist[0][0] = 0
    pq = [(0, 0, 0)]  # (cost, city, mode)
    
    while pq:
        cost, city, mode = heapq.heappop(pq)
        if cost > dist[city][mode]:
            continue
        for next_city in range(N):
            if next_city == city:
                continue
            if mode == 0:  # by company car
                next_cost = cost + D[city][next_city] * A
                if next_cost < dist[next_city][0]:
                    dist[next_city][0] = next_cost
                    heapq.heappush(pq, (next_cost, next_city, 0))
                next_cost = cost + D[city][next_city] * B + C
                if next_cost < dist[next_city][1]:
                    dist[next_city][1] = next_cost
                    heapq.heappush(pq, (next_cost, next_city, 1))
            else:  # by train
                next_cost = cost + D[city][next_city] * B + C
                if next_cost < dist[next_city][1]:
                    dist[next_city][1] = next_cost
                    heapq.heappush(pq, (next_cost, next_city, 1))
    return min(dist[N-1])

N, A, B, C = map(int, input().split())
D = [list(map(int, input().split())) for _ in range(N)]

print(dijkstra(N, A, B, C, D))