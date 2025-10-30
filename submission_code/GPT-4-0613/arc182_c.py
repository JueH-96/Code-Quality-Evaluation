MOD = 998244353
N, M = map(int, input().split())
 
fact = [1]
for i in range(1, M+1):
    fact.append(fact[-1]*i%MOD)
 
inv = [1, 1]
inv_fact = [1, 1]
for i in range(2, M+1):
    inv.append((MOD - MOD//i) * inv[MOD%i] % MOD)
    inv_fact.append(inv_fact[-1]*inv[-1]%MOD)
 
def comb(n, r):
    if n < r or r < 0:
        return 0
    return fact[n] * inv_fact[r] * inv_fact[n-r] % MOD
 
def f(x):
    res = 0
    for i in range(1, M+1):
        res += comb(M, i) * pow(i, x, MOD)
        res %= MOD
    return res
 
ans = 0
for i in range(1, N+1):
    ans += f(i)
    ans %= MOD
print(ans)