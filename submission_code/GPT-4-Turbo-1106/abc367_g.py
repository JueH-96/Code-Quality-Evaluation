MOD = 998244353

def power(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def solve(N, M, K, A):
    if M == 1:
        xor_sum = 0
        for a in A:
            xor_sum ^= a
        return power(xor_sum, K, MOD)
    
    dp = [0] * M
    dp[0] = 1
    for a in A:
        next_dp = [0] * M
        for i in range(M):
            next_dp[i] = (dp[i] + dp[(i - 1) % M]) % MOD
        dp = next_dp
        dp[0] = (dp[0] + power(2, N - 1, MOD)) % MOD
    
    ans = dp[0] - 1
    for a in A:
        ans = (ans - power(a, K, MOD)) % MOD
    
    return ans

N, M, K = map(int, input().split())
A = list(map(int, input().split()))

print(solve(N, M, K, A))