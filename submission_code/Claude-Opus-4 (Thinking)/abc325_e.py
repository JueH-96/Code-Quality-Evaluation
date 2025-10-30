import heapq

def dijkstra_car(d, n, a, start):
    INF = 10**18
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > dist[u]:
            continue
        
        for v in range(n):
            if u != v:
                new_dist = dist[u] + d[u][v] * a
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    
    return dist

def dijkstra_train(d, n, b, c, start):
    INF = 10**18
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > dist[u]:
            continue
        
        for v in range(n):
            if u != v:
                new_dist = dist[u] + d[u][v] * b + c
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    
    return dist

# Read input
n, a, b, c = map(int, input().split())
d = []
for i in range(n):
    row = list(map(int, input().split()))
    d.append(row)

# Dijkstra from city 0 using car
car_dist = dijkstra_car(d, n, a, 0)

# Dijkstra from city n-1 using train
train_dist = dijkstra_train(d, n, b, c, n-1)

# Find minimum time
min_time = 10**18
for i in range(n):
    # Switch at city i
    total_time = car_dist[i] + train_dist[i]
    min_time = min(min_time, total_time)

print(min_time)