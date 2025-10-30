def main():
    import sys
    sys.setrecursionlimit(200000)
    input = sys.stdin.readline
    
    N = int(input().strip())
    tree = [[] for _ in range(N)]
    for _ in range(N - 1):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        tree[a].append(b)
        tree[b].append(a)
    C = list(map(int, input().split()))
    
    # subtree_sum[u] will hold the sum of C values in the subtree of u (including u)
    subtree_sum = [0] * N  
    # dp[u] holds the sum_{v in subtree u} (distance from u to v) * C[v]
    dp = [0] * N  
    
    # First DFS: Compute dp for a fixed root=0 and subtree_sum.
    def dfs1(u, parent):
        subtree_sum[u] = C[u]
        for v in tree[u]:
            if v == parent:
                continue
            dfs1(v, u)
            subtree_sum[u] += subtree_sum[v]
            dp[u] += dp[v] + subtree_sum[v]
    
    dfs1(0, -1)
    
    # f[u] will denote the result f(u)
    f = [0] * N
    f[0] = dp[0]
    total_C = sum(C)
    
    # Second DFS: re-rooting, from u to v: f(v) = f(u) + (total_C - 2 * subtree_sum[v])
    def dfs2(u, parent):
        for v in tree[u]:
            if v == parent:
                continue
            f[v] = f[u] + (total_C - 2 * subtree_sum[v])
            dfs2(v, u)
    
    dfs2(0, -1)
    
    print(min(f))
    
if __name__ == '__main__':
    main()