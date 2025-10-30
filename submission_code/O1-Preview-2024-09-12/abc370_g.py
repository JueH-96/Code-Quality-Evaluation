MOD = 998244353

def modinv(a):
    return pow(a, MOD - 2, MOD)

N, M = map(int, input().split())
N_mod = N % MOD
total = pow(N_mod, M, MOD)
ans = total * modinv(2) % MOD
print(ans)