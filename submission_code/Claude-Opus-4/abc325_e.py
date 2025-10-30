import heapq

# Read input
N, A, B, C = map(int, input().split())
D = []
for i in range(N):
    D.append(list(map(int, input().split())))

# Initialize distances
# dist[i][0] = minimum time to reach city i using only car
# dist[i][1] = minimum time to reach city i after switching to train
INF = float('inf')
dist = [[INF, INF] for _ in range(N)]
dist[0][0] = 0  # Start at city 1 (index 0) with car

# Priority queue: (time, city, mode)
# mode: 0 = car, 1 = train
pq = [(0, 0, 0)]

while pq:
    time, city, mode = heapq.heappop(pq)
    
    # Skip if we've found a better path
    if time > dist[city][mode]:
        continue
    
    # Try all neighboring cities
    for next_city in range(N):
        if city == next_city:
            continue
        
        if mode == 0:  # Currently using car
            # Continue with car
            car_time = time + D[city][next_city] * A
            if car_time < dist[next_city][0]:
                dist[next_city][0] = car_time
                heapq.heappush(pq, (car_time, next_city, 0))
            
            # Switch to train
            train_time = time + D[city][next_city] * B + C
            if train_time < dist[next_city][1]:
                dist[next_city][1] = train_time
                heapq.heappush(pq, (train_time, next_city, 1))
        
        else:  # Currently using train (mode == 1)
            # Can only continue with train
            train_time = time + D[city][next_city] * B + C
            if train_time < dist[next_city][1]:
                dist[next_city][1] = train_time
                heapq.heappush(pq, (train_time, next_city, 1))

# The answer is the minimum time to reach city N (index N-1)
print(min(dist[N-1][0], dist[N-1][1]))