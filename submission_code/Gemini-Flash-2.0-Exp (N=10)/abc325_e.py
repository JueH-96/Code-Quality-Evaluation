import heapq

def solve():
    n, a, b, c = map(int, input().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, input().split())))

    dist_car = [float('inf')] * n
    dist_car[0] = 0
    pq_car = [(0, 0)]

    while pq_car:
        cost, u = heapq.heappop(pq_car)
        if cost > dist_car[u]:
            continue
        for v in range(n):
            if d[u][v] > 0:
                new_cost = cost + d[u][v] * a
                if new_cost < dist_car[v]:
                    dist_car[v] = new_cost
                    heapq.heappush(pq_car, (new_cost, v))
    
    
    dist_train = [float('inf')] * n
    dist_train[0] = dist_car[0]
    pq_train = [(dist_car[0], 0)]

    while pq_train:
        cost, u = heapq.heappop(pq_train)
        if cost > dist_train[u]:
            continue
        for v in range(n):
            if d[u][v] > 0:
                new_cost = cost + d[u][v] * b + c
                if new_cost < dist_train[v]:
                    dist_train[v] = new_cost
                    heapq.heappush(pq_train, (new_cost, v))

    print(dist_train[n-1])

solve()