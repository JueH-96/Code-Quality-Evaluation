import heapq

def solve():
    n, a, b, c = map(int, input().split())
    d_matrix = []
    for _ in range(n):
        d_matrix.append(list(map(int, input().split())))
    
    min_time = [[float('inf')] * (n + 1) for _ in range(2)]
    min_time[0][1] = 0
    
    pq = [(0, 0, 1)] # (time, mode, city), mode 0: car, 1: train
    
    while pq:
        current_time, current_mode, current_city = heapq.heappop(pq)
        if current_time > min_time[current_mode][current_city]:
            continue
            
        for next_city in range(1, n + 1):
            if next_city == current_city:
                continue
                
            if current_mode == 0: # currently in car mode
                # Option 1: continue with car
                travel_time_car = d_matrix[current_city-1][next_city-1] * a
                new_time_car = current_time + travel_time_car
                if new_time_car < min_time[0][next_city]:
                    min_time[0][next_city] = new_time_car
                    heapq.heappush(pq, (new_time_car, 0, next_city))
                    
                # Option 2: switch to train
                travel_time_train = d_matrix[current_city-1][next_city-1] * b + c
                new_time_train = current_time + travel_time_train
                if new_time_train < min_time[1][next_city]:
                    min_time[1][next_city] = new_time_train
                    heapq.heappush(pq, (new_time_train, 1, next_city))
                    
            elif current_mode == 1: # already in train mode
                travel_time_train = d_matrix[current_city-1][next_city-1] * b + c
                new_time_train = current_time + travel_time_train
                if new_time_train < min_time[1][next_city]:
                    min_time[1][next_city] = new_time_train
                    heapq.heappush(pq, (new_time_train, 1, next_city))
                    
    result = min(min_time[0][n], min_time[1][n])
    print(result)

if __name__ == '__main__':
    solve()