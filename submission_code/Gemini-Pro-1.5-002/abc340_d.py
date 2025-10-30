# YOUR CODE HERE
import heapq

def solve():
    n = int(input())
    stages = []
    for _ in range(n - 1):
        a, b, x = map(int, input().split())
        stages.append((a, b, x))

    dist = [float('inf')] * n
    dist[0] = 0
    pq = [(0, 0)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        if u == n - 1:
            print(d)
            return

        if u < n - 1:
            if dist[u + 1] > d + stages[u][0]:
                dist[u + 1] = d + stages[u][0]
                heapq.heappush(pq, (dist[u + 1], u + 1))
            
            next_stage = stages[u][2] - 1
            if dist[next_stage] > d + stages[u][1]:
                dist[next_stage] = d + stages[u][1]
                heapq.heappush(pq, (dist[next_stage], next_stage))

solve()