def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(1, n + 1):
        dp[i] = (dp[i-1] * 2) % mod
        current_sum = 0
        for j in range(i - 1, -1, -1):
            current_sum += a[j]
            if current_sum == k:
                dp[i] = (dp[i] - dp[j]) % mod
                break
    
    print(dp[n])

solve()