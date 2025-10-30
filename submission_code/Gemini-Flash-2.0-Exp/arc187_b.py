def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    
    q = b.count(-1)
    
    total_f = 0
    
    for i in range(m**q):
        temp = i
        curr_b = []
        neg_idx = 0
        
        for val in b:
            if val == -1:
                curr_b.append(temp % m + 1)
                temp //= m
            else:
                curr_b.append(val)
        
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if curr_b[u] <= curr_b[v]:
                    adj[u].append(v)
                    adj[v].append(u)
        
        visited = [False] * n
        num_components = 0
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
        
        for u in range(n):
            if not visited[u]:
                dfs(u)
                num_components += 1
        
        total_f = (total_f + num_components) % 998244353
    
    print(total_f)

solve()