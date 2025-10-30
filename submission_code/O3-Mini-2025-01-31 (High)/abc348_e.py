def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx])
    idx += 1
    graph = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(data[idx])
        v = int(data[idx + 1])
        idx += 2
        graph[u].append(v)
        graph[v].append(u)
    # Read C values (1-indexed)
    C = [0] * (n + 1)
    for i in range(1, n + 1):
        C[i] = int(data[idx])
        idx += 1

    # We'll compute two arrays using DFS:
    # subtree_sum[v] = sum of C[u] for all nodes u in the subtree of v (including v)
    # dp[v] = sum_{u in subtree(v)} (C[u] * distance(v, u))
    subtree_sum = [0] * (n + 1)
    dp = [0] * (n + 1)
    
    def dfs1(v, parent):
        subtree_sum[v] = C[v]
        dp[v] = 0
        for child in graph[v]:
            if child == parent:
                continue
            dfs1(child, v)
            subtree_sum[v] += subtree_sum[child]
            dp[v] += dp[child] + subtree_sum[child]
    
    dfs1(1, 0)
    total = subtree_sum[1]  # total sum of all C values
    
    # Use a re-rooting DFS to compute f(v) for every vertex v.
    # f(v) is defined as sum_{i=1}^{N} (C[i] * d(v, i)).
    # We already have f(1) = dp[1].
    answer = dp[1]
    f = [0] * (n + 1)
    f[1] = dp[1]
    
    def dfs2(v, parent):
        nonlocal answer
        for child in graph[v]:
            if child == parent:
                continue
            # When re-rooting from v to child:
            # f(child) = f(v) + (total - 2 * subtree_sum[child]).
            f[child] = f[v] + total - 2 * subtree_sum[child]
            if f[child] < answer:
                answer = f[child]
            dfs2(child, v)
    
    dfs2(1, 0)
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()