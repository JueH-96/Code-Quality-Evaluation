import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
C = int(data[3])

D = []
index = 4
for i in range(N):
    D.append(list(map(int, data[index:index+N])))
    index += N

def dijkstra(N, D, A, B, C):
    # We need two sets of distances because we can switch from car to train
    # but not vice versa. So we track the minimum time to each city by car and by train.
    inf = float('inf')
    min_time_car = [inf] * N
    min_time_train = [inf] * N
    min_time_car[0] = 0  # Start at city 1 with car
    
    # Priority queue of tuples (current_time, city_index, is_train)
    # is_train is False if using car, True if using train
    pq = [(0, 0, False)]  # (time, city, is_train)
    heapq.heapify(pq)
    
    while pq:
        current_time, u, is_train = heapq.heappop(pq)
        
        if is_train:
            if current_time > min_time_train[u]:
                continue
        else:
            if current_time > min_time_car[u]:
                continue
        
        # Explore neighbors
        for v in range(N):
            if u == v:
                continue
            travel_time = D[u][v]
            
            # If currently using car, consider both staying with car and switching to train
            if not is_train:
                # Continue with car
                new_time_car = current_time + travel_time * A
                if new_time_car < min_time_car[v]:
                    min_time_car[v] = new_time_car
                    heapq.heappush(pq, (new_time_car, v, False))
                
                # Switch to train
                new_time_train = current_time + travel_time * B + C
                if new_time_train < min_time_train[v]:
                    min_time_train[v] = new_time_train
                    heapq.heappush(pq, (new_time_train, v, True))
            else:
                # Continue with train
                new_time_train = current_time + travel_time * B + C
                if new_time_train < min_time_train[v]:
                    min_time_train[v] = new_time_train
                    heapq.heappush(pq, (new_time_train, v, True))
    
    # We need the minimum time to reach city N
    return min(min_time_car[N-1], min_time_train[N-1])

result = dijkstra(N, D, A, B, C)
print(result)