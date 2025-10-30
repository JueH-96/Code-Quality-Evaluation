def solve():
    n = int(input())
    a = [0] + list(map(int, input().split()))
    
    MOD = 998244353
    
    # Build tree where parent of i is a[i]
    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        children[a[i]].append(i)
    
    # Sort children by position
    for i in range(n + 1):
        children[i].sort()
    
    # Precompute factorials and inverse factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    def C(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
    
    # DP on tree
    size = [0] * (n + 1)
    
    def dfs(u):
        size[u] = 1
        result = 1
        
        # For each child, we need to handle the positions between u and the child
        prev = u
        for v in children[u]:
            # Positions between prev and v must have larger values than u
            between = v - prev - 1
            
            # Recursively solve for child
            child_result = dfs(v)
            result = result * child_result % MOD
            
            # Choose positions for the 'between' elements
            result = result * C(size[u] - 1 + between, between) % MOD
            
            size[u] += size[v] + between
            prev = v
        
        return result
    
    # Handle root's children
    ans = 1
    total_size = 0
    
    for v in children[0]:
        child_result = dfs(v)
        ans = ans * child_result % MOD
        ans = ans * C(total_size + size[v], size[v]) % MOD
        total_size += size[v]
    
    print(ans)

solve()