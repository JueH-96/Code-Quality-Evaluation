def main():
    import sys, heapq
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    X = int(next(it))
    # Build two adjacency lists:
    # graph[v] : all vertices u such that there is a forward edge v -> u.
    graph = [[] for _ in range(n+1)]
    # rev_graph[v] : all vertices u such that there is an edge u -> v in the input.
    rev_graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        rev_graph[v].append(u)
    
    # We'll use Dijkstra's algorithm on an augmented state space.
    # Each state is represented as (vertex, parity) where:
    #  - parity 0 means the graph is in the original orientation.
    #  - parity 1 means all edges have been reversed.
    #
    # When in state (v,0) we can move along any edge in graph[v] (cost 1).
    # When in state (v,1) we can move along any edge in rev_graph[v] (cost 1) because the reversed graph
    # has edges v -> u when originally u -> v.
    #
    # Also, from any state (v, p) we can perform a reversal operation that toggles the state
    # to (v, 1-p) at a cost of X.
    
    INF = float('inf')
    # dist[v][p] holds the minimum cost to reach vertex v with orientation parity p.
    dist = [[INF, INF] for _ in range(n+1)]
    dist[1][0] = 0  # starting at vertex 1 in original orientation
    
    # Priority queue: (cost, vertex, parity)
    hq = []
    heapq.heappush(hq, (0, 1, 0))
    
    while hq:
        d, v, p = heapq.heappop(hq)
        # If we already found a better way to state (v, p), skip.
        if d != dist[v][p]:
            continue
        # If we have reached the goal vertex (in any orientation), we can output the answer.
        if v == n:
            sys.stdout.write(str(d))
            return
        
        # Option 1: move along an edge (cost + 1).
        if p == 0:
            # In original state, take forward edges v -> u.
            for nv in graph[v]:
                nd = d + 1
                if nd < dist[nv][p]:
                    dist[nv][p] = nd
                    heapq.heappush(hq, (nd, nv, p))
        else:
            # In reversed state, the allowed moves (v -> u) come
            # from original edges u -> v.
            for nv in rev_graph[v]:
                nd = d + 1
                if nd < dist[nv][p]:
                    dist[nv][p] = nd
                    heapq.heappush(hq, (nd, nv, p))
        
        # Option 2: reverse the orientation of all edges (flip parity), cost + X.
        new_parity = 1 - p
        nd = d + X
        if nd < dist[v][new_parity]:
            dist[v][new_parity] = nd
            heapq.heappush(hq, (nd, v, new_parity))
    
    # If the loop ends without early return (it should not, since a path exists),
    # we output the minimum cost found for vertex n among both states.
    ans = min(dist[n][0], dist[n][1])
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()