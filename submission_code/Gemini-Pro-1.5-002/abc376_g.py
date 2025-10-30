def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[p[i]].append(i + 1)
        
    def power(x, y):
        res = 1
        x = x % mod
        while y > 0:
            if y & 1:
                res = (res * x) % mod
            y = y >> 1
            x = (x * x) % mod
        return res
    
    def inv(n):
        return power(n, mod - 2)

    total_a = sum(a)
    inv_total_a = inv(total_a)
    
    expected_ops = 0
    
    for i in range(1, n + 1):
        path = []
        curr = i
        while curr != 0:
            path.append(curr)
            curr = p[curr-1]
        path.append(0)
        path.reverse()
        
        prob_i = (a[i-1] * inv_total_a) % mod
        
        curr_expected = 0
        for j in range(1, len(path)):
            subtree_sum = 0
            q = [path[j]]
            while q:
                u = q.pop(0)
                if u <= n:
                    subtree_sum = (subtree_sum + a[u-1]) % mod
                for v in adj[u-1 if u > 0 else u]:
                    q.append(v)
            
            curr_expected = (curr_expected + (subtree_sum * inv_total_a) % mod) % mod
            
        expected_ops = (expected_ops + (prob_i * curr_expected) % mod) % mod
        
    print(expected_ops)

t = int(input())
for _ in range(t):
    solve()