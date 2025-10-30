MOD = 998244353

def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))

    q = b.count(-1)
    total_f = 0

    for i in range(m**q):
        temp_b = list(b)
        temp_i = i
        for j in range(n):
            if temp_b[j] == -1:
                temp_b[j] = (temp_i % m) + 1
                temp_i //= m

        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if temp_b[u] <= temp_b[v]:
                    adj[u].append(v)
                    adj[v].append(u)

        visited = [False] * n
        num_connected_components = 0

        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)

        for u in range(n):
            if not visited[u]:
                dfs(u)
                num_connected_components += 1
        total_f = (total_f + num_connected_components) % MOD

    print(total_f)

solve()