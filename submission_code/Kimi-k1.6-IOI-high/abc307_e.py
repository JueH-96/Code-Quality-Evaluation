MOD = 998244353

N, M = map(int, input().split())

if M == 1:
    print(0)
else:
    term1 = pow(M - 1, N, MOD)
    if N % 2 == 0:
        term2 = (M - 1) % MOD
    else:
        term2 = (M - 1) * (MOD - 1) % MOD
    ans = (term1 + term2) % MOD
    print(ans)