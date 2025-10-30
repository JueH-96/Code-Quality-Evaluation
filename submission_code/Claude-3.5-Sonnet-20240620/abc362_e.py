# YOUR CODE HERE
MOD = 998244353

def nCr(n, r):
    if r > n:
        return 0
    res = 1
    for i in range(r):
        res = (res * (n - i)) % MOD
        res = (res * pow(i + 1, MOD - 2, MOD)) % MOD
    return res

N = int(input())
A = list(map(int, input().split()))

dp = [[[0] * N for _ in range(N)] for _ in range(N + 1)]

for i in range(N):
    for j in range(i + 1, N):
        dp[2][i][j] = 1

for k in range(3, N + 1):
    for i in range(N):
        for j in range(i + 1, N):
            diff = A[j] - A[i]
            next_val = A[j] + diff
            for l in range(j + 1, N):
                if A[l] == next_val:
                    dp[k][i][j] = (dp[k][i][j] + dp[k-1][j][l]) % MOD

result = [0] * (N + 1)
result[1] = N

for k in range(2, N + 1):
    for i in range(N):
        for j in range(i + 1, N):
            result[k] = (result[k] + dp[k][i][j]) % MOD

print(*result[1:])