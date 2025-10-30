import sys

MOD = 998244353

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # binomial coefficients C(K, t)
    binom = [1] * (K + 1)
    for t in range(1, K + 1):
        binom[t] = binom[t - 1] * (K - t + 1) * pow(t, MOD - 2, MOD) % MOD

    cum = [0] * (K + 1)
    cum[0] = 1                     # S_0^0 = 1
    ans = 0
    S = 0

    for a in A:
        S = (S + a) % MOD

        # powers of current prefix sum
        powS = [1] * (K + 1)
        for p in range(1, K + 1):
            powS[p] = powS[p - 1] * S % MOD

        for t in range(K + 1):
            coef = binom[t]
            if t & 1:                          # multiply by (-1)^t
                coef = (MOD - coef) % MOD
            add = coef * powS[K - t] % MOD * cum[t] % MOD
            ans += add
            if ans >= MOD:
                ans -= MOD

        for t in range(K + 1):
            cum[t] += powS[t]
            if cum[t] >= MOD:
                cum[t] -= MOD

    print(ans % MOD)

if __name__ == "__main__":
    main()