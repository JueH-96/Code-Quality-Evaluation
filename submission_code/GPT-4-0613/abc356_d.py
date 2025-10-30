MOD = 998244353
MAX = 61
fact = [1]*MAX
inv = [1]*MAX
invfact = [1]*MAX
for i in range(1,MAX):
    fact[i] = (fact[i-1]*i)%MOD
    inv[i] = MOD - inv[MOD%i]*(MOD//i)%MOD
    invfact[i] = invfact[i-1]*inv[i]%MOD

def C(n, r):
    if r < 0 or r > n:
        return 0
    return fact[n]*invfact[r]*invfact[n-r]%MOD

def solve(N, M):
    dp = [[0]*MAX for _ in range(MAX)]
    dp[60][0] = 1
    for b in range(59, -1, -1):
        for c in range(MAX):
            for d in [0, 1]:
                if (N>>b&1) < d:
                    continue
                for e in [0, 1]:
                    if (M>>b&1) < e:
                        continue
                    dp[b][c+d] += dp[b+1][c]
                    dp[b][c+d] %= MOD
    ans = 0
    for c in range(MAX):
        ans += dp[0][c]*fact[c]
        ans %= MOD
    for c in range(1, MAX):
        ans += C(N, c)*pow(2, N-c, MOD)*fact[c]
        ans %= MOD
    return ans

N, M = map(int, input().split())
print(solve(N, M))