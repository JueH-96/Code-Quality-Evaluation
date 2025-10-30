# YOUR CODE HERE
MOD = 998244353

def solve():
    K = int(input())
    C = list(map(int, input().split()))
    
    dp = [0] * (K + 1)
    dp[0] = 1
    
    for c in C:
        new_dp = [0] * (K + 1)
        for i in range(K + 1):
            for j in range(min(c, K - i) + 1):
                new_dp[i + j] = (new_dp[i + j] + dp[i]) % MOD
        dp = new_dp
    
    print(sum(dp[1:]) % MOD)

solve()