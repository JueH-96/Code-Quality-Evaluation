def solve():
    n, q = map(int, input().split())
    edges = []
    for i in range(q):
        l, r, c = map(int, input().split())
        edges.append((l, r, c, n + i + 1))

    import heapq
    
    adj = [[] for _ in range(n + q + 1)]
    for l, r, c, node in edges:
        for j in range(l, r + 1):
            adj[node].append((j, c))
            adj[j].append((node, c))

    def mst(n, adj):
        visited = [False] * (n + q + 1)
        pq = [(0, 1)]  # (cost, node)
        total_cost = 0
        num_visited = 0

        while pq:
            cost, u = heapq.heappop(pq)
            
            if visited[u]:
                continue
            
            visited[u] = True
            total_cost += cost
            num_visited += 1

            for v, weight in adj[u]:
                if not visited[v]:
                    heapq.heappush(pq, (weight, v))

        if num_visited == n + q:
            return total_cost
        else:
            return -1

    result = mst(n, adj)
    print(result)

solve()