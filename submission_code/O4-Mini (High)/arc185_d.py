def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline
    mod = 998244353

    N, M = map(int, input().split())
    # Special case N=1: just a single path of length M, cover time = 2*M
    if N == 1:
        print((2 * M) % mod)
        return

    # Precompute inverses up to N for harmonic numbers
    inv = [0] * (N + 1)
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = (mod - mod // i) * inv[mod % i] % mod

    # Compute H_N = 1 + 1/2 + ... + 1/N mod
    HN = 0
    for i in range(1, N + 1):
        HN = (HN + inv[i]) % mod

    # A1 = A[1] = 2*M - 1 mod
    A1 = (2 * M - 1) % mod
    # A_M = A[M] = M^2 mod
    AM = (M * M) % mod

    # ans = 2*N*M*A1*H_N - 2*N*A_M  (mod)
    term1 = (2 * N) % mod
    term1 = (term1 * (M % mod)) % mod
    term1 = (term1 * A1) % mod
    term1 = (term1 * HN) % mod

    term2 = (2 * N) % mod
    term2 = (term2 * AM) % mod

    ans = (term1 - term2) % mod
    print(ans)

if __name__ == "__main__":
    main()