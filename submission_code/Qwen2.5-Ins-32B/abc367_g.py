# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def mod_pow(x, n, mod=MOD):
    res = 1
    while n > 0:
        if n & 1:
            res = res * x % mod
        x = x * x % mod
        n >>= 1
    return res

def solve(N, M, K, A):
    cnt = [0] * (1 << 20)
    for a in A:
        cnt[a] += 1

    dp = [[0] * M for _ in range(1 << 20)]
    dp[0][0] = 1
    for i in range(1 << 20):
        for j in range(M):
            dp[i][j] = dp[i][j] * mod_pow(2, cnt[i], MOD) % MOD
            if i > 0:
                dp[i][j] += dp[i ^ (i & -i)][j - 1]
                dp[i][j] %= MOD

    ans = 0
    for i in range(1 << 20):
        if M <= cnt[i]:
            ans += mod_pow(i, K) * dp[i][M - 1] % MOD
            ans %= MOD
    return ans

def main():
    data = list(map(int, input().split()))
    N, M, K = data[:3]
    A = data[3:]
    print(solve(N, M, K, A))

main()