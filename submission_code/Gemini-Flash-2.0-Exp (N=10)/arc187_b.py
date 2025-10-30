def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    
    neg_one_indices = [i for i, x in enumerate(b) if x == -1]
    num_neg_ones = len(neg_one_indices)
    
    total_components = 0
    mod = 998244353
    
    for i in range(m ** num_neg_ones):
        temp_b = b[:]
        temp_val = i
        for idx in neg_one_indices:
            temp_b[idx] = (temp_val % m) + 1
            temp_val //= m
        
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if temp_b[u] <= temp_b[v]:
                    adj[u].append(v)
                    adj[v].append(u)
        
        visited = [False] * n
        components = 0
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
        
        for u in range(n):
            if not visited[u]:
                dfs(u)
                components += 1
        
        total_components = (total_components + components) % mod
        
    print(total_components)

solve()