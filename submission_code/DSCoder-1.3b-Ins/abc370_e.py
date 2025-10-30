import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    mod = 998244353
    inv2 = pow(2, mod-2, mod)
    fact = [1]
    inv = [1]
    for i in range(1, N+1):
        fact.append(fact[-1] * i % mod)
        inv.append((inv[-1] * inv2) % mod)

    ans = 0
    for i in range(1, N+1):
        fact_inv = fact[i] * inv[N-i] % mod
        ans = (ans + fact_inv * pow(K+1, N-i, mod) * pow(i, mod-2, mod)) % mod

    print(ans)

solve()