def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    dp = {}

    def get_dp(mask, current_node):
        return dp.get((mask, current_node), float('inf'))

    for i in range(n):
        dp[(1 << i, i)] = 0

    for mask_size in range(1, n + 1):
        for mask in range(1 << n):
            if bin(mask).count('1') == mask_size:
                for current_node in range(n):
                    if (mask >> current_node) & 1:
                        if mask_size == 1:
                            continue
                        prev_mask = mask ^ (1 << current_node)
                        for prev_node in range(n):
                            if (prev_mask >> prev_node) & 1:
                                if get_dp(prev_mask, prev_node) != float('inf') and dist[prev_node][current_node] != float('inf'):
                                    dp[(mask, current_node)] = min(get_dp(mask, current_node), get_dp(prev_mask, prev_node) + dist[prev_node][current_node])
                        if mask_size == n:
                            for next_node in range(n):
                                if dist[current_node][next_node] != float('inf'):
                                    dp[(mask | (1 << next_node), next_node)] = min(get_dp(mask | (1 << next_node), next_node), get_dp(mask, current_node) + dist[current_node][next_node])

    min_cost = float('inf')
    final_mask = (1 << n) - 1
    for i in range(n):
        min_cost = min(min_cost, get_dp(final_mask, i))

    if min_cost == float('inf'):
        print("No")
    else:
        print(min_cost)

solve()