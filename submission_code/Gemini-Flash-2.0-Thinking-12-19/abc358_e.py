def solve():
    k = int(input())
    c = list(map(int, input().split()))
    mod = 998244353
    
    max_val = k
    fact = [1] * (max_val + 1)
    inv_fact = [1] * (max_val + 1)
    
    for i in range(1, max_val + 1):
        fact[i] = (fact[i-1] * i) % mod
        
    inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
    for i in range(max_val - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i + 1)) % mod
        
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n-r]) % mod
        return (num * den) % mod
        
    dp = [[0] * (k + 1) for _ in range(27)]
    
    for j in range(k + 1):
        if j <= c[0]:
            dp[1][j] = 1
        else:
            dp[1][j] = 0
            
    for i in range(2, 27):
        limit_c = c[i-1]
        for j in range(k + 1):
            for l in range(min(j, limit_c) + 1):
                term = (nCr_mod(j, l) * dp[i-1][j-l]) % mod
                dp[i][j] = (dp[i][j] + term) % mod
                
    total_count = 0
    for length in range(1, k + 1):
        total_count = (total_count + dp[26][length]) % mod
        
    print(total_count)

if __name__ == '__main__':
    solve()