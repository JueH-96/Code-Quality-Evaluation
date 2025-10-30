def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    
    q = b.count(-1)
    mod = 998244353
    
    ans = 0
    
    for i in range(m**q):
        temp = i
        curr_b = b[:]
        for j in range(n):
            if curr_b[j] == -1:
                curr_b[j] = (temp % m) + 1
                temp //= m
        
        adj = [[] for _ in range(n)]
        for j in range(n):
            for k in range(j + 1, n):
                if curr_b[j] <= curr_b[k]:
                    adj[j].append(k)
                    adj[k].append(j)
        
        visited = [False] * n
        count = 0
        
        for j in range(n):
            if not visited[j]:
                count += 1
                stack = [j]
                visited[j] = True
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
        
        ans = (ans + count) % mod
        
    print(ans)

solve()