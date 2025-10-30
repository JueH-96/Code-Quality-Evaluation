def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    inv_n = pow(n, mod - 2, mod)
    
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        dp[i] = (sum(dp[:i]) * inv_n + a[i-1] + sum(a[:i-1]) * inv_n) % mod
        dp[i] = (dp[i] * n) % mod
        dp[i] = (dp[i] * pow(n - (i - 1), mod - 2, mod)) % mod
        
    
    print(sum(dp[1:]) * inv_n % mod)

solve()