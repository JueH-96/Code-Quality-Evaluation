def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = int(input_data[1])
    B = int(input_data[2])
    C = int(input_data[3])

    # Read distance matrix
    D = []
    idx = 4
    for i in range(N):
        row = list(map(int, input_data[idx:idx+N]))
        idx += N
        D.append(row)

    # We will use a Dijkstra approach with two "modes" per city:
    # mode 0: traveling by car
    # mode 1: traveling by train
    #
    # dist_car[i]   = minimum cost to reach city i+1 in car mode
    # dist_train[i] = minimum cost to reach city i+1 in train mode
    #
    # State (cost, mode, city_index).
    # We start from (0, 0, 0) meaning cost=0, mode=car, city=1 (index=0).
    # Adjacencies:
    #  - From car(i): 
    #       to car(j) at cost = D[i][j] * A
    #       to train(i) at cost = 0 (switch modes in same city)
    #  - From train(i):
    #       to train(j) at cost = D[i][j] * B + C

    INF = 10**20
    dist_car = [INF] * N
    dist_train = [INF] * N

    dist_car[0] = 0  # Start at city 1 by car
    pq = []
    # push initial state: (cost=0, mode=car, city=0)
    heapq.heappush(pq, (0, 0, 0))

    visited_car = [False]*N
    visited_train = [False]*N

    while pq:
        current_cost, mode, city = heapq.heappop(pq)

        if mode == 0:
            # If we've already found a better route for car(city), skip
            if current_cost > dist_car[city]:
                continue
            visited_car[city] = True
        else:
            # If we've already found a better route for train(city), skip
            if current_cost > dist_train[city]:
                continue
            visited_train[city] = True

        # If we've reached city N (index N-1) in either mode with best cost, we can stop
        # But we can't be sure there's not a cheaper train route still in the queue,
        # so let's not forcibly stop unless we want an optimization. We'll keep going.
        # (It's safe to do a standard Dijkstra loop though.)

        if city == N-1:
            # We've reached city N in some mode. We could break, as
            # this is a valid shortest route for that mode.
            # But let's not break prematurely; it is still a normal approach to keep going.
            pass

        if mode == 0:
            # We are in car mode at city 'city'
            # 1) We can switch to train in the same city (cost 0)
            if not visited_train[city]:
                new_cost = current_cost
                if new_cost < dist_train[city]:
                    dist_train[city] = new_cost
                    heapq.heappush(pq, (new_cost, 1, city))

            # 2) Move by car to other cities
            for nxt in range(N):
                if nxt == city:
                    continue
                if not visited_car[nxt]:
                    travel_cost = D[city][nxt] * A
                    new_cost = current_cost + travel_cost
                    if new_cost < dist_car[nxt]:
                        dist_car[nxt] = new_cost
                        heapq.heappush(pq, (new_cost, 0, nxt))
        else:
            # We are in train mode at city 'city'
            # Move by train to other cities
            for nxt in range(N):
                if nxt == city:
                    continue
                if not visited_train[nxt]:
                    travel_cost = D[city][nxt] * B + C
                    new_cost = current_cost + travel_cost
                    if new_cost < dist_train[nxt]:
                        dist_train[nxt] = new_cost
                        heapq.heappush(pq, (new_cost, 1, nxt))

    # The answer is the minimum of reaching city N by car or by train
    ans = min(dist_car[N-1], dist_train[N-1])
    print(ans)

if __name__ == "__main__":
    main()