def solve():
    n = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    children = [[] for _ in range(n + 1)]
    for i in range(n):
        children[parents[i]].append(i + 1)
    
    total_sum = sum(a)
    inv_total_sum = pow(total_sum, mod - 2, mod)
    
    probs = [(a[i] * inv_total_sum) % mod for i in range(n)]
    
    subtree_probs = [0] * (n + 1)
    subtree_sizes = [0] * (n + 1)
    
    def dfs(u):
        subtree_probs[u] = probs[u-1] if u > 0 else 0
        subtree_sizes[u] = 1
        
        for v in children[u]:
            dfs(v)
            subtree_probs[u] = (subtree_probs[u] + subtree_probs[v]) % mod
            subtree_sizes[u] += subtree_sizes[v]
    
    dfs(0)
    
    q = []
    for child in children[0]:
        q.append(child)
    
    q.sort(key=lambda x: subtree_probs[x] * pow(subtree_sizes[x], mod-2, mod) % mod, reverse=True)
    
    ans = 0
    searched_count = 1
    
    for u in q:
        ans = (ans + searched_count * subtree_probs[u]) % mod
        searched_count = (searched_count + subtree_sizes[u]) % mod
        
    print(ans)

t = int(input())
for _ in range(t):
    solve()