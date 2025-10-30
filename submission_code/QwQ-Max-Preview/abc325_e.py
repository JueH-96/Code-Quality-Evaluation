import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = int(input[idx])
    idx += 1
    B = int(input[idx])
    idx += 1
    C = int(input[idx])
    idx += 1
    
    D = []
    for _ in range(N):
        row = list(map(int, input[idx:idx+N]))
        idx += N
        D.append(row)
    
    INF = float('inf')
    car_dist = [INF] * (N + 1)
    train_dist = [INF] * (N + 1)
    car_dist[1] = 0
    train_dist[1] = INF  # Initially, switching to train is allowed but needs to be processed
    
    heap = []
    heapq.heappush(heap, (0, 1, 'car'))
    
    while heap:
        current_cost, current_city, current_mode = heapq.heappop(heap)
        if current_mode == 'car' and current_cost > car_dist[current_city]:
            continue
        if current_mode == 'train' and current_cost > train_dist[current_city]:
            continue
        
        if current_mode == 'car':
            # Transition to other cities via car
            for j in range(1, N + 1):
                if j == current_city:
                    continue
                new_cost = current_cost + D[current_city - 1][j - 1] * A
                if new_cost < car_dist[j]:
                    car_dist[j] = new_cost
                    heapq.heappush(heap, (new_cost, j, 'car'))
            # Switch to train at current_city
            if current_cost < train_dist[current_city]:
                train_dist[current_city] = current_cost
                heapq.heappush(heap, (current_cost, current_city, 'train'))
        else:
            # Transition to other cities via train
            for j in range(1, N + 1):
                if j == current_city:
                    continue
                new_cost = current_cost + D[current_city - 1][j - 1] * B + C
                if new_cost < train_dist[j]:
                    train_dist[j] = new_cost
                    heapq.heappush(heap, (new_cost, j, 'train'))
    
    result = min(car_dist[N], train_dist[N])
    print(result)

if __name__ == "__main__":
    main()