def main():
    import sys
    sys.setrecursionlimit(10**7)

    MOD = 998244353

    # Read inputs
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    # ----------------------------------------------------------------
    # Derivation (informal summary):
    #
    # 1) The given graph is a "star of N chains," each chain having M
    #    new (unpainted) vertices plus the common root 0 in the center.
    #    - Vertex 0 is connected to each of {1, 2, ..., N}.
    #    - Then for each c in [1..N], we have a chain c -> c+N -> c+2N -> ... -> c+(M-1)*N.
    #
    # 2) It turns out (and can be proved by careful random-walk analysis)
    #    that the expected time (number of moves) to cover (paint) all
    #    vertices, starting from 0 (already painted), is:
    #
    #       E = M^2 × ( 2N × H_N  −  1 )
    #
    #    where H_N = 1 + 1/2 + 1/3 + ... + 1/N is the N-th harmonic number.
    #
    # 3) Since we must output this expected value modulo 998244353, we
    #    compute everything in modular arithmetic, taking care to use
    #    modular inverses for 1/k.
    #
    # ----------------------------------------------------------------

    # 1) Precompute inverses 1..N in O(N) time using the standard trick:
    inv = [0] * (N+1)
    inv[1] = 1
    for i in range(2, N+1):
        # inv[i] = pow(i, MOD-2, MOD)  # could do this, but it's O(log MOD) each
        # Instead use the O(N) method:
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD

    # 2) Compute H_N = sum_{k=1..N} (1/k)  modulo MOD
    HN = 0
    for i in range(1, N+1):
        HN = (HN + inv[i]) % MOD

    # 3) Compute A = 2*N * HN - 1  (mod MOD)
    twoN = (2 * N) % MOD
    A = (twoN * HN) % MOD
    A = (A - 1) % MOD  # subtract 1 in mod

    # 4) Compute M^2 mod
    M_sq = (M % MOD) * (M % MOD) % MOD

    # 5) Final answer = M^2 * A  (mod MOD)
    ans = (M_sq * A) % MOD

    print(ans)

# Do not forget to call main!
main()