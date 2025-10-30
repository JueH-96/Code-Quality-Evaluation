def main():
    import sys, heapq
    input_data = sys.stdin.read().split()
    itr = iter(input_data)
    N = int(next(itr))
    A = int(next(itr))
    B = int(next(itr))
    C = int(next(itr))
    
    # read distance matrix D (0-indexed)
    D = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            D[i][j] = int(next(itr))
    
    # We will use Dijkstra's algorithm twice.
    # In the first pass, we compute the minimum travel time from city 1 (node 0)
    # to every other city using only company car. In car mode,
    # the travel time from city i to j is D[i][j] * A.
    #
    # In the second pass, we compute the minimum travel time from every city
    # (as starting point in the train segment) to city N (node N-1) in train mode.
    # In train mode, the travel time from city i to j is D[i][j] * B + C.
    # Since switching to train can be done only once (and is irreversible),
    # any journey is composed of a car segment from city 1 to some city i,
    # and then a train segment from city i to city N.
    # Hence, the total time using a switch at city i is:
    #   car_cost[i] + train_cost[i].
    #
    # The answer is the minimum among these sums
    # (including the possibility of not switching at all, i.e. i == N).
    
    def dijkstra(start, mode):
        """
        start: starting node index (0-indexed)
        mode: 'car' or 'train'
          - In 'car' mode, weight = D[u][v] * A.
          - In 'train' mode, weight = D[u][v] * B + C.
        Returns the list of minimal costs from the start node to every other node.
        """
        INF = 10**18
        dist = [INF] * N
        dist[start] = 0
        # (current_cost, node)
        heap = [(0, start)]
        while heap:
            cur, u = heapq.heappop(heap)
            if cur != dist[u]:
                continue
            for v in range(N):
                if u == v:
                    continue
                if mode == 'car':
                    nxt = cur + D[u][v] * A
                else:  # mode == 'train'
                    nxt = cur + D[u][v] * B + C
                if nxt < dist[v]:
                    dist[v] = nxt
                    heapq.heappush(heap, (nxt, v))
        return dist
    
    # Compute minimal times from city 1 using company car.
    car_cost = dijkstra(0, 'car')
    
    # Compute minimal times from city N using train mode.
    # Instead of running dijkstra from every possible city to N,
    # we run it backwards by treating the problem as undirected
    # (which it is, because D is symmetric) and using train weights.
    train_cost = dijkstra(N - 1, 'train')
    
    result = 10**18
    for i in range(N):
        total = car_cost[i] + train_cost[i]
        if total < result:
            result = total
    
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()