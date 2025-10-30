import sys
import heapq

def minimum_travel_time(N, A, B, C, D):
    # Initialize distances for both modes of transport
    car_time = [float('inf')] * N
    train_time = [float('inf')] * N
    
    # Starting point
    car_time[0] = 0  # Starting in city 1 with car
    train_time[0] = float('inf')  # Cannot start with train
    
    # Priority queue for Dijkstra's algorithm
    pq = []
    heapq.heappush(pq, (0, 0, 'car'))  # (time, city_index, mode)
    
    while pq:
        current_time, city, mode = heapq.heappop(pq)
        
        # If we reached city N-1 (city N in 1-indexed)
        if city == N - 1:
            return current_time
        
        # If we are in car mode
        if mode == 'car':
            # Explore all cities from current city
            for next_city in range(N):
                if next_city != city:
                    travel_time = D[city][next_city] * A
                    new_time = current_time + travel_time
                    if new_time < car_time[next_city]:
                        car_time[next_city] = new_time
                        heapq.heappush(pq, (new_time, next_city, 'car'))
                    
                    # Switch to train mode
                    train_travel_time = D[city][next_city] * B + C
                    new_time_train = current_time + train_travel_time
                    if new_time_train < train_time[next_city]:
                        train_time[next_city] = new_time_train
                        heapq.heappush(pq, (new_time_train, next_city, 'train'))
        
        # If we are in train mode
        elif mode == 'train':
            # Explore all cities from current city
            for next_city in range(N):
                if next_city != city:
                    travel_time = D[city][next_city] * B + C
                    new_time = current_time + travel_time
                    if new_time < train_time[next_city]:
                        train_time[next_city] = new_time
                        heapq.heappush(pq, (new_time, next_city, 'train'))
    
    # The answer is the minimum time to reach city N-1
    return min(car_time[N-1], train_time[N-1])

def main():
    input_data = sys.stdin.read().strip().splitlines()
    first_line = list(map(int, input_data[0].split()))
    N, A, B, C = first_line
    D = [list(map(int, line.split())) for line in input_data[1:]]
    
    result = minimum_travel_time(N, A, B, C, D)
    print(result)

if __name__ == "__main__":
    main()