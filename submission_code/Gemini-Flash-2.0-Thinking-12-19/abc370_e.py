def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + a[i]
        
    dp = [0] * (n + 1)
    dp[0] = 1
    
    mod = 998244353
    
    for i in range(1, n + 1):
        for j in range(i):
            subsequence_sum = prefix_sums[i] - prefix_sums[j]
            if subsequence_sum != k:
                dp[i] = (dp[i] + dp[j]) % mod
                
    print(dp[n])

if __name__ == '__main__':
    solve()