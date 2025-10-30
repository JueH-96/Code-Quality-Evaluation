def solve():
    n, q = map(int, input().split())
    adj = {}
    isolated_vertices = n

    def count_isolated():
        count = 0
        for i in range(1, n + 1):
            if i not in adj or len(adj.get(i, set())) == 0:
                count += 1
        return count

    for _ in range(q):
        query = list(map(int, input().split()))
        query_type = query[0]

        if query_type == 1:
            u, v = query[1], query[2]

            u_isolated_before = u not in adj or len(adj.get(u, set())) == 0
            v_isolated_before = v not in adj or len(adj.get(v, set())) == 0

            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()

            if v not in adj[u]:
                adj[u].add(v)
                if u_isolated_before:
                    isolated_vertices -= 1
            if u not in adj[v]:
                adj[v].add(u)
                if v_isolated_before:
                    isolated_vertices -= 1

        elif query_type == 2:
            v_remove = query[1]
            if v_remove in adj:
                if len(adj[v_remove]) > 0:
                    isolated_vertices += 1
                neighbors = list(adj[v_remove])
                for neighbor in neighbors:
                    adj[neighbor].discard(v_remove)
                    if len(adj[neighbor]) == 0:
                        isolated_vertices += 1
                del adj[v_remove]

        print(count_isolated())

solve()