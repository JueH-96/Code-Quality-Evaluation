def solve():
    a, b, m = map(int, input().split())
    n = a * b - 1
    
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i-1] * i) % m
    
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], m - 2, m)
    for i in range(n - 1, 0, -1):
        inv_fact[i] = (inv_fact[i+1] * (i + 1)) % m

    def ncr(n, r):
        if r < 0 or r > n:
            return 0
        return (fact[n] * inv_fact[r] * inv_fact[n-r]) % m

    dp = {}
    def count_permutations(i, j, k):
        if (i, j, k) in dp:
            return dp[(i, j, k)]
        
        if i == 0 and j == 0:
            return 1 if k == 0 else 0
        
        if i == 0:
            return ncr(k - 1, j - 1)
        
        if j == 0:
            return ncr(k - 1, i - 1)
            
        res = 0
        for x in range(i):
            res = (res + count_permutations(x, j - 1, k - 1) * ncr(k - 1, i - x -1)) % m
        
        dp[(i, j, k)] = res
        return res
    
    ans = 0
    for k in range(n + 1):
        ans = (ans + count_permutations(a - 1, b - 1, k) * (k + 1 - a) * (k + 1 - b) * ncr(n, k)) % m
    
    print(ans)

solve()