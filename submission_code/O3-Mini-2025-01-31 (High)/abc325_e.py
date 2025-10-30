import sys, heapq

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))
    
    # Read the distance matrix
    mat = [[int(next(it)) for _ in range(n)] for _ in range(n)]
    
    INF = float('inf')
    # dist_car[i]: minimum time to reach city i in car mode (state 0)
    # dist_train[i]: minimum time to reach city i in train mode (state 1)
    dist_car = [INF] * n
    dist_train = [INF] * n
    dist_car[0] = 0  # start in car mode at city 1 (index 0)
    
    # Priority queue items: (current_time, city, mode)
    # mode == 0 for car; mode == 1 for train.
    pq = [(0, 0, 0)]
    
    while pq:
        time, city, mode = heapq.heappop(pq)
        if mode == 0:
            if time != dist_car[city]:
                continue
        else:
            if time != dist_train[city]:
                continue
        
        current_row = mat[city]
        if mode == 0:
            base = time
            # From car mode, we have two kinds of moves:
            # 1) Continue with the company car.
            # 2) Switch to train (which is allowed in any city) and take a train leg.
            for j in range(n):
                if j == city:
                    continue
                # Option 1: Travel by company car from city to j
                new_time = base + A * current_row[j]
                if new_time < dist_car[j]:
                    dist_car[j] = new_time
                    heapq.heappush(pq, (new_time, j, 0))
                # Option 2: Switch to train and travel from city to j using train.
                new_time2 = base + B * current_row[j] + C
                if new_time2 < dist_train[j]:
                    dist_train[j] = new_time2
                    heapq.heappush(pq, (new_time2, j, 1))
        else:  # mode == 1 (already in train mode)
            base = time
            # In train mode you can only travel by train.
            for j in range(n):
                if j == city:
                    continue
                new_time = base + B * current_row[j] + C
                if new_time < dist_train[j]:
                    dist_train[j] = new_time
                    heapq.heappush(pq, (new_time, j, 1))
    
    # The destination is city N (index n-1). Answer is the minimum time reaching city N
    answer = min(dist_car[n - 1], dist_train[n - 1])
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()