import heapq

def solve():
    N, A, B, C = map(int, input().split())
    D = []
    for i in range(N):
        row = list(map(int, input().split()))
        D.append(row)
    
    # dist[i][0] = minimum time to reach city i using car
    # dist[i][1] = minimum time to reach city i using train
    dist = [[float('inf')] * 2 for _ in range(N)]
    
    # Start at city 0 (0-indexed) with car
    dist[0][0] = 0
    
    # Priority queue: (time, city, mode) where mode 0=car, 1=train
    pq = [(0, 0, 0)]
    
    while pq:
        time, city, mode = heapq.heappop(pq)
        
        if time > dist[city][mode]:
            continue
            
        # Try going to all other cities
        for next_city in range(N):
            if next_city == city:
                continue
                
            if mode == 0:  # Currently using car
                # Option 1: Continue with car
                car_time = time + D[city][next_city] * A
                if car_time < dist[next_city][0]:
                    dist[next_city][0] = car_time
                    heapq.heappush(pq, (car_time, next_city, 0))
                
                # Option 2: Switch to train
                train_time = time + D[city][next_city] * B + C
                if train_time < dist[next_city][1]:
                    dist[next_city][1] = train_time
                    heapq.heappush(pq, (train_time, next_city, 1))
                    
            else:  # Currently using train
                # Can only continue with train
                train_time = time + D[city][next_city] * B + C
                if train_time < dist[next_city][1]:
                    dist[next_city][1] = train_time
                    heapq.heappush(pq, (train_time, next_city, 1))
    
    # Answer is minimum time to reach city N-1 (0-indexed) using either mode
    result = min(dist[N-1][0], dist[N-1][1])
    print(result)

solve()