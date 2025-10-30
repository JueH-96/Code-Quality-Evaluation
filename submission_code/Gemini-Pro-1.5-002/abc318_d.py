# YOUR CODE HERE
def solve():
    n = int(input())
    d = []
    for _ in range(n - 1):
        d.append(list(map(int, input().split())))

    adj = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj[i][j] = adj[j][i] = d[i][j - i - 1]

    ans = 0
    for i in range(1 << (n * (n - 1) // 2)):
        edges = []
        k = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (i >> k) & 1:
                    edges.append((u, v))
                k += 1

        valid = True
        used = [False] * n
        for u, v in edges:
            if used[u] or used[v]:
                valid = False
                break
            used[u] = True
            used[v] = True
        
        if valid:
            current_weight = 0
            for u, v in edges:
                current_weight += adj[u][v]
            ans = max(ans, current_weight)

    print(ans)

solve()