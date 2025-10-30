import heapq

def solve():
    n, a, b, c = map(int, input().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, input().split())))

    dist_car = [float('inf')] * n
    dist_train = [float('inf')] * n
    dist_car[0] = 0

    pq = [(0, 0, 0)]  # (time, city, mode) where 0 is car, 1 is train

    while pq:
        time, city, mode = heapq.heappop(pq)

        if mode == 0:
            if time > dist_car[city]:
                continue
            # Travel by car
            for neighbor in range(n):
                travel_time = d[city][neighbor] * a
                if dist_car[city] + travel_time < dist_car[neighbor]:
                    dist_car[neighbor] = dist_car[city] + travel_time
                    heapq.heappush(pq, (dist_car[neighbor], neighbor, 0))
            # Switch to train
            if dist_car[city] < dist_train[city]:
                dist_train[city] = dist_car[city]
                heapq.heappush(pq, (dist_train[city], city, 1))

        elif mode == 1:
            if time > dist_train[city]:
                continue
            # Travel by train
            for neighbor in range(n):
                travel_time = d[city][neighbor] * b + c
                if dist_train[city] + travel_time < dist_train[neighbor]:
                    dist_train[neighbor] = dist_train[city] + travel_time
                    heapq.heappush(pq, (dist_train[neighbor], neighbor, 1))

    print(min(dist_car[n - 1], dist_train[n - 1]))

solve()