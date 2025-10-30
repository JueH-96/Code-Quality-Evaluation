def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, K = map(int, input().split())

    # When N = 1, black ball never moves; expected position is 1.
    if N == 1:
        print(1)
        return

    # Precompute modular inverses and constants
    inv2 = (MOD + 1) // 2
    invN = pow(N, MOD - 2, MOD)

    # a = 1 - 2/N  mod MOD
    # but 1 - 2/N = (N - 2)/N
    a = (N - 2) * invN % MOD

    # a^K mod MOD
    aK = pow(a, K, MOD)

    # Expected value: E = ((N+1) - (N-1)*a^K) / 2
    # Do it in mod arithmetic
    term = ( (N + 1) - ( (N - 1) * aK ) % MOD ) % MOD
    ans = term * inv2 % MOD

    print(ans)

if __name__ == "__main__":
    main()