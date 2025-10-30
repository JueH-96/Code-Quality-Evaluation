from collections import defaultdict
from math import gcd

MOD = 998244353

def modinv(x):
    return pow(x, MOD - 2, MOD)

def main():
    N = int(input())
    P = [tuple(map(int, input().split())) for _ in range(N - 1)]
    P = [(p - 1, q - 1) for p, q in P]

    # Union-Find
    par = list(range(N))
    siz = [1] * N

    def find(x):
        if par[x] == x:
            return x
        else:
            par[x] = find(par[x])
            return par[x]

    def unite(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return
        par[x] = y
        siz[y] += siz[x]

    # Probability of winning
    win = [1] * N
    for p, q in P:
        a = siz[find(p)]
        b = siz[find(q)]
        win[p] *= a / (a + b)
        win[q] *= b / (a + b)
        win[p] %= MOD
        win[q] %= MOD
        unite(p, q)

    # Expected number of wins
    exp = [0] * N
    for i in range(N):
        for j in range(N):
            if find(j) == find(i):
                exp[i] += win[j]
        exp[i] = int(exp[i] * modinv(siz[find(i)]) % MOD)

    print(*exp)

main()