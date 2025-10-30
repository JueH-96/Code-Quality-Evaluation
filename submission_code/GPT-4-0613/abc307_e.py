MOD = 998244353
N, M = map(int, input().split())
 
fact = [1]*(N+1)
inv = [1]*(N+1)
 
for i in range(1, N+1):
    fact[i] = (fact[i-1]*i)%MOD
    inv[i] = pow(fact[i], MOD-2, MOD)
 
def C(n, r):
    if r < 0 or r > n:
        return 0
    return fact[n]*inv[r]*inv[n-r]%MOD
 
ans = M*pow(M-1, N-1, MOD)
if N > 1:
    ans -= M*C(N, 2)*pow(M-2, N-2, MOD)
print(ans%MOD)