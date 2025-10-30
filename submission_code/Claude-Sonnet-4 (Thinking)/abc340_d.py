import heapq

n = int(input())
edges = []
for i in range(1, n):
    a, b, x = map(int, input().split())
    edges.append((a, b, x))

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
    
    if stage < n:
        # Option 1: spend A_{stage} to go to stage+1
        a, b, x = edges[stage - 1]
        if dist[stage] + a < dist[stage + 1]:
            dist[stage + 1] = dist[stage] + a
            heapq.heappush(pq, (dist[stage + 1], stage + 1))
        
        # Option 2: spend B_{stage} to go to stage X_{stage}
        if dist[stage] + b < dist[x]:
            dist[x] = dist[stage] + b
            heapq.heappush(pq, (dist[x], x))