MOD = 998244353

N, M = map(int, input().split())

if M == 1:
    print(0)
else:
    a = M - 1
    term1 = pow(a, N, MOD)
    term2 = (pow(-1, N, MOD) * a) % MOD
    ans = (term1 + term2) % MOD
    print(ans)