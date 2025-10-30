def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, K = map(int, input().split())
    # Precompute modular inverses we need
    inv2 = (MOD + 1) // 2
    invN = pow(N, MOD - 2, MOD)

    # r = (N-2)/N mod
    r = (N - 2) % MOD * invN % MOD

    # r^K mod
    rk = pow(r, K, MOD)

    # m_K = ( (N+1) - (N-1)*r^K )/2  mod
    term1 = (N + 1) % MOD
    term2 = (N - 1) % MOD * rk % MOD
    res = (term1 - term2) % MOD
    res = res * inv2 % MOD

    print(res)


if __name__ == "__main__":
    main()