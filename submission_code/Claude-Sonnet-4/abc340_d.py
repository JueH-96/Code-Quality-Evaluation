import heapq

n = int(input())
graph = [[] for _ in range(n + 1)]

for i in range(1, n):
    a, b, x = map(int, input().split())
    graph[i].append((i + 1, a))  # Go to next stage with cost a
    graph[i].append((x, b))      # Go to stage x with cost b

# Dijkstra's algorithm
dist = [float('inf')] * (n + 1)
dist[1] = 0
pq = [(0, 1)]  # (distance, stage)

while pq:
    d, stage = heapq.heappop(pq)
    
    if d > dist[stage]:
        continue
    
    if stage == n:
        print(d)
        break
    
    for next_stage, cost in graph[stage]:
        new_dist = d + cost
        if new_dist < dist[next_stage]:
            dist[next_stage] = new_dist
            heapq.heappush(pq, (new_dist, next_stage))