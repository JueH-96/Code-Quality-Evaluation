def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + a[i]
    
    dp = [0] * (n + 1)
    dp[0] = 1
    mod = 998244353
    
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            current_sum = prefix_sum[i] - prefix_sum[j-1]
            if current_sum != k:
                dp[i] = (dp[i] + dp[j-1]) % mod
                
    print(dp[n])

if __name__ == '__main__':
    solve()