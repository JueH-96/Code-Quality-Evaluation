def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        adj[u].append((v, w))
        adj[v].append((u, w))
    # We will DFS from 1 to N
    visited = [False] * (N+1)
    INF = 1 << 62
    best = [INF]  # use list for mutability in nested function

    def dfs(u, cur_xor):
        # If we reached N, update best answer
        if u == N:
            if cur_xor < best[0]:
                best[0] = cur_xor
            return
        # Otherwise, continue DFS
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                dfs(v, cur_xor ^ w)
                visited[v] = False

    # start DFS
    visited[1] = True
    dfs(1, 0)
    # print result
    print(best[0])

if __name__ == "__main__":
    main()