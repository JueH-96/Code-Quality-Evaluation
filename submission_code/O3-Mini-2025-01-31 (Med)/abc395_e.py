def main():
    import sys, heapq
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    X = int(next(it))

    # Build the graph: for each vertex, store its forward neighbors and also its reverse neighbors.
    # When the graph is reversed (parity=1), we can move by using the reverse neighbors.
    forward = [[] for _ in range(n + 1)]
    reverse = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        forward[u].append(v)
        reverse[v].append(u)

    # We'll use Dijkstra's algorithm on a state space of (vertex, parity)
    # Parity 0: graph is in original orientation.
    # Parity 1: graph is reversed.
    # Moves:
    #   - Move along an edge: cost 1, use "forward" if parity==0 or "reverse" if parity==1.
    #   - Reverse the entire graph: cost X, toggles the parity without changing the vertex.
    INF = 10**18
    dist = [[INF, INF] for _ in range(n + 1)]
    # Starting at vertex 1 with parity 0.
    dist[1][0] = 0
    pq = [(0, 1, 0)]  # (cost, vertex, parity)

    while pq:
        cur_cost, vertex, parity = heapq.heappop(pq)
        if cur_cost != dist[vertex][parity]:
            continue

        # Operation 1: move along an edge (cost 1).
        if parity == 0:
            # If graph is in original orientation, use forward edges
            for nxt in forward[vertex]:
                new_cost = cur_cost + 1
                if new_cost < dist[nxt][parity]:
                    dist[nxt][parity] = new_cost
                    heapq.heappush(pq, (new_cost, nxt, parity))
        else:
            # If graph is reversed, use reverse neighbors.
            for nxt in reverse[vertex]:
                new_cost = cur_cost + 1
                if new_cost < dist[nxt][parity]:
                    dist[nxt][parity] = new_cost
                    heapq.heappush(pq, (new_cost, nxt, parity))

        # Operation 2: reverse the entire graph (toggle parity, cost X).
        new_cost = cur_cost + X
        new_parity = 1 - parity
        if new_cost < dist[vertex][new_parity]:
            dist[vertex][new_parity] = new_cost
            heapq.heappush(pq, (new_cost, vertex, new_parity))

    # The answer is the minimum cost to reach vertex n in either orientation.
    answer = min(dist[n][0], dist[n][1])
    sys.stdout.write(str(answer))


if __name__ == '__main__':
    main()