def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N_K_idx = 0
    N = int(data[N_K_idx])
    K = int(data[N_K_idx + 1])
    N_K_idx += 2
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        a = int(data[N_K_idx]) - 1
        b = int(data[N_K_idx + 1]) - 1
        N_K_idx += 2
        adj[a].append(b)
        adj[b].append(a)
    
    # Mark vertices in S
    S = set()
    for _ in range(K):
        v = int(data[N_K_idx]) - 1
        N_K_idx += 1
        S.add(v)
    
    # DFS to compute count[u]
    def dfs_count(u, p):
        count = 1 if u in S else 0
        for v in adj[u]:
            if v != p:
                count += dfs_count(v, u)
        return count
    
    # DFS to mark vertices in minimal subtree
    def dfs_mark(u, p, mark):
        if u in S:
            mark[u] = True
        for v in adj[u]:
            if v != p:
                child_mark = dfs_mark(v, u, mark)
                if child_mark:
                    mark[u] = True
        return mark[u]
    
    # Perform DFS to compute count[u]
    count = [0] * N
    count[0] = dfs_count(0, -1)
    
    # Perform DFS to mark vertices in minimal subtree
    mark = [False] * N
    dfs_mark(0, -1, mark)
    
    # Count the number of marked vertices
    result = sum(mark)
    print(result)

if __name__ == '__main__':
    main()