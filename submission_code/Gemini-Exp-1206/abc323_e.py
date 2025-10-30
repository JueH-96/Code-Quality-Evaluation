def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    mod = 998244353
    
    dp = [0] * (x + 1)
    dp[0] = 1
    
    inv_n = pow(n, mod - 2, mod)
    
    for i in range(x):
        for j in range(n):
            if i + t[j] <= x:
                dp[i + t[j]] = (dp[i + t[j]] + dp[i] * inv_n) % mod
    
    ans = 0
    for i in range(max(0, x - t[0] + 1), x + 1):
        ans = (ans + dp[i] * inv_n) % mod
        
    print(ans)

solve()