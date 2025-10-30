def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    mod = 998244353
    
    dp = [0] * (x + 1)
    dp[0] = 1
    
    for _ in range(x):
        new_dp = [0] * (x + 1)
        for j in range(x):
            if dp[j] > 0:
                for k in range(n):
                    if j + t[k] <= x:
                        new_dp[j + t[k]] = (new_dp[j + t[k]] + dp[j] * pow(n, mod-2, mod)) % mod
        dp = new_dp
    
    ans = 0
    if t[0] <= x:
      ans = dp[x-t[0]] * pow(n, mod-2, mod) % mod
    
    print(ans)

solve()