# YOUR CODE HERE
MOD = 998244353

def pow_mod(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def inv_mod(a, m):
    return pow_mod(a, m - 2, m)

N, X = map(int, input().split())
T = list(map(int, input().split()))

total_len = sum(T)
prob_list = [t * inv_mod(total_len, MOD) % MOD for t in T]

dp = {}
dp[0] = [0] * N
dp[0][0] = 1

for i in range(1, X + 2):
    dp[i] = [0] * N
    for j in range(N):
        for k in range(N):
            if i - T[k] >= 0:
                dp[i][j] = (dp[i][j] + dp[i - T[k]][k] * prob_list[k]) % MOD

ans = dp[X + 1][0]
print(ans)