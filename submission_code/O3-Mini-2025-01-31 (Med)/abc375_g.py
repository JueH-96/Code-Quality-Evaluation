def main():
    import sys, heapq
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # List of roads and graph creation (undirected)
    roads = []
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        c = int(next(it))
        roads.append((a, b, c))
        graph[a].append((b, c))
        graph[b].append((a, c))

    INF = 10**20

    # Compute shortest distances from 1: d1[u] = distance from node 1 to u.
    d1 = [INF]*(n+1)
    d1[1] = 0
    heap = [(0, 1)]
    while heap:
        dist, u = heapq.heappop(heap)
        if dist != d1[u]:
            continue
        for v, w in graph[u]:
            nd = dist + w
            if nd < d1[v]:
                d1[v] = nd
                heapq.heappush(heap, (nd, v))

    # Compute shortest distances from n: dN[u] = distance from u to node n.
    dN = [INF]*(n+1)
    dN[n] = 0
    heap = [(0, n)]
    while heap:
        dist, u = heapq.heappop(heap)
        if dist != dN[u]:
            continue
        for v, w in graph[u]:
            nd = dist + w
            if nd < dN[v]:
                dN[v] = nd
                heapq.heappush(heap, (nd, v))
    
    # D is the shortest distance from 1 to n (exists by assumption)
    D = d1[n]

    # Build the directed acyclic graph (DAG) of edges that can appear on a shortest path.
    # For an undirected edge (a,b,c), if d1[a] + c == d1[b],
    # then the edge from a -> b lies on a shortest path. Similarly for b -> a.
    dag = [[] for _ in range(n+1)]
    rev_dag = [[] for _ in range(n+1)]
    for a, b, c in roads:
        if d1[a] + c == d1[b]:
            dag[a].append(b)
            rev_dag[b].append(a)
        if d1[b] + c == d1[a]:
            dag[b].append(a)
            rev_dag[a].append(b)
    
    # Compute dp1[u]: number of shortest paths from 1 to u in the DAG.
    dp1 = [0]*(n+1)
    dp1[1] = 1
    # Process vertices in increasing order of d1.
    order = list(range(1, n+1))
    order.sort(key=lambda x: d1[x])
    for u in order:
        if dp1[u] == 0:
            continue
        for v in dag[u]:
            dp1[v] += dp1[u]
    
    total_paths = dp1[n]  # Total number of shortest paths from 1 to n.
    
    # Compute dp2[u]: number of shortest paths from u to n in the DAG.
    dp2 = [0]*(n+1)
    dp2[n] = 1
    for u in order[::-1]:
        if dp2[u] == 0:
            continue
        for v in rev_dag[u]:
            dp2[v] += dp2[u]

    # For each road, decide if its removal changes the shortest distance.
    # The idea is:
    #   If the road is not used in any shortest path, then removing it will not affect the shortest distance.
    #   If the road is used in some shortest path, check if every shortest path uses it.
    #   For an edge used in one orientation (say from u to v – note d1[u] < d1[v] given weights are positive),
    #   the number of shortest paths using that edge is dp1[u] * dp2[v].
    #   If dp1[u] * dp2[v] equals total_paths, then every shortest path goes through that edge.
    #   In that case, removing the edge will change the distance.
    #
    # Also, if one case is reachable and after removal it becomes unreachable, it is considered different.
    
    out_lines = []
    for a, b, c in roads:
        used_critically = False
        # Check if the road can lie on a shortest path in either orientation.
        if d1[a] + c + dN[b] == D:
            # In this case, a -> b is a candidate; note that with positive weights d1[a] < d1[b]
            if dp1[a] * dp2[b] == total_paths:
                used_critically = True
        elif d1[b] + c + dN[a] == D:
            if dp1[b] * dp2[a] == total_paths:
                used_critically = True
        out_lines.append("Yes" if used_critically else "No")
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()