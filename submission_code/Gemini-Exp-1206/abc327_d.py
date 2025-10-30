def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    adj = [[] for _ in range(n + 1)]
    for i in range(m):
        adj[a[i]].append(b[i])
        adj[b[i]].append(a[i])

    colors = [-1] * (n + 1)

    def is_bipartite(start_node):
        q = [(start_node, 0)]
        colors[start_node] = 0

        while q:
            u, color = q.pop(0)
            for v in adj[u]:
                if colors[v] == -1:
                    colors[v] = 1 - color
                    q.append((v, 1 - color))
                elif colors[v] == color:
                    return False
        return True

    for i in range(1, n + 1):
        if colors[i] == -1:
            if not is_bipartite(i):
                print("No")
                return

    print("Yes")

solve()