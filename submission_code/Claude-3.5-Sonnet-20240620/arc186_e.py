# YOUR CODE HERE
MOD = 998244353

def pow_mod(x, n):
    result = 1
    while n > 0:
        if n & 1:
            result = (result * x) % MOD
        x = (x * x) % MOD
        n >>= 1
    return result

def solve(N, M, K, X):
    dp = [[[0] * (1 << M) for _ in range(M + 1)] for _ in range(N + 1)]
    dp[0][0][0] = 1

    for i in range(N):
        for j in range(M + 1):
            for mask in range(1 << M):
                if dp[i][j][mask] == 0:
                    continue
                
                for k in range(1, K + 1):
                    new_j = j
                    new_mask = mask
                    for m in range(M):
                        if (mask & (1 << m)) == 0 and X[m] == k:
                            new_j = m + 1
                            new_mask |= (1 << m)
                            break
                    
                    dp[i + 1][new_j][new_mask] = (dp[i + 1][new_j][new_mask] + dp[i][j][mask]) % MOD

    total = pow_mod(K, N)
    invalid = dp[N][M][(1 << M) - 1]
    result = (total - invalid + MOD) % MOD

    return result

N, M, K = map(int, input().split())
X = list(map(int, input().split()))

answer = solve(N, M, K, X)
print(answer)