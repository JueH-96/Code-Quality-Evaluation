import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    # Read input values
    index = 0
    N = int(data[index])
    index += 1
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    
    D = []
    for i in range(N):
        D.append([int(data[index + j]) for j in range(N)])
        index += N
    
    # Initialize distances
    dist_car = [float('inf')] * N
    dist_train = [float('inf')] * N
    
    # Start from city 1
    dist_car[0] = 0
    
    # Priority queue for Dijkstra's algorithm
    pq = []
    heapq.heappush(pq, (0, 0, 'car'))  # (time, city, mode)
    
    while pq:
        current_time, current_city, current_mode = heapq.heappop(pq)
        
        if current_mode == 'car' and current_time > dist_car[current_city]:
            continue
        if current_mode == 'train' and current_time > dist_train[current_city]:
            continue
        
        for next_city in range(N):
            if next_city == current_city:
                continue
            
            travel_time_car = D[current_city][next_city] * A
            travel_time_train = D[current_city][next_city] * B + C
            
            # If currently in car, we can continue by car
            if current_mode == 'car':
                if current_time + travel_time_car < dist_car[next_city]:
                    dist_car[next_city] = current_time + travel_time_car
                    heapq.heappush(pq, (dist_car[next_city], next_city, 'car'))
                
                # Or switch to train
                if current_time + travel_time_train < dist_train[next_city]:
                    dist_train[next_city] = current_time + travel_time_train
                    heapq.heappush(pq, (dist_train[next_city], next_city, 'train'))
            
            # If currently in train, we can only continue by train
            if current_mode == 'train':
                if current_time + travel_time_train < dist_train[next_city]:
                    dist_train[next_city] = current_time + travel_time_train
                    heapq.heappush(pq, (dist_train[next_city], next_city, 'train'))
    
    # The answer is the minimum time to reach city N (index N-1)
    result = min(dist_car[N-1], dist_train[N-1])
    print(result)