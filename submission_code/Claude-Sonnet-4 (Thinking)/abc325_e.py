import heapq

n, a, b, c = map(int, input().split())
d = []
for i in range(n):
    row = list(map(int, input().split()))
    d.append(row)

# dist[city][mode] = minimum time to reach city with mode (0=car, 1=train)
dist = [[float('inf')] * 2 for _ in range(n)]
dist[0][0] = 0  # Start at city 0 with car

# Priority queue: (distance, city, mode)
pq = [(0, 0, 0)]

while pq:
    curr_dist, city, mode = heapq.heappop(pq)
    
    if curr_dist > dist[city][mode]:
        continue
    
    if mode == 0:  # Currently using car
        # Option 1: Switch to train at the same city (no cost)
        if dist[city][1] > curr_dist:
            dist[city][1] = curr_dist
            heapq.heappush(pq, (curr_dist, city, 1))
        
        # Option 2: Travel to other cities
        for next_city in range(n):
            if next_city != city:
                # Continue with car
                cost = d[city][next_city] * a
                if dist[next_city][0] > curr_dist + cost:
                    dist[next_city][0] = curr_dist + cost
                    heapq.heappush(pq, (curr_dist + cost, next_city, 0))
                
                # Switch to train and travel
                cost = d[city][next_city] * b + c
                if dist[next_city][1] > curr_dist + cost:
                    dist[next_city][1] = curr_dist + cost
                    heapq.heappush(pq, (curr_dist + cost, next_city, 1))
    
    else:  # Currently using train
        # Can only continue with train
        for next_city in range(n):
            if next_city != city:
                cost = d[city][next_city] * b + c
                if dist[next_city][1] > curr_dist + cost:
                    dist[next_city][1] = curr_dist + cost
                    heapq.heappush(pq, (curr_dist + cost, next_city, 1))

# Answer is the minimum time to reach the last city with either mode
print(min(dist[n-1][0], dist[n-1][1]))