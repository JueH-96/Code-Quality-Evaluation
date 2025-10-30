def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data)
    MOD = 998244353

    # If N=1, the graph is just a single chain (path) of length M+1,
    # and the expected cover time for a path of length (M+1), starting at one end,
    # is M^2. (Well-known result for a simple random walk on a path.)
    if N == 1:
        # Just output (M^2) mod
        ans = (M % MOD) * (M % MOD) % MOD
        print(ans)
        return

    # For general N >= 2, the tree can be viewed as N "spokes" (branches),
    # each of length M, emanating from the central (painted) vertex 0.
    #
    # A known (and can be proven) closed-form for the expected cover time on this
    # "star of chains" graph is:
    #
    #    E(N, M) = (2*N * H_{N-1} + 1) * (M^2),
    #
    # where H_{n} = 1 + 1/2 + 1/3 + ... + 1/n is the n-th harmonic number.
    #
    # In modular arithmetic, we must compute the harmonic number by summing
    # modular inverses 1/k for k=1..(N-1). Then multiply by 2*N, add 1,
    # and finally multiply by M^2, all mod 998244353.

    # Precompute inverses up to N-1 in O(N) time using the standard technique:
    # inv[i] = - (mod // i) * inv[mod % i] (mod)  (plus the necessary mod adjustments).
    # This relies on 998244353 being prime and is a well-known trick.
    inv = [0]*(N)
    inv[1] = 1
    for i in range(2, N):
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD

    # Compute the (N-1)-th harmonic number modulo MOD: H_{N-1}
    H = 0
    for i in range(1, N):
        H = (H + inv[i]) % MOD

    # factor = (2*N * H + 1) mod
    # then multiply by M^2 mod
    factor = ( (2*N) % MOD ) * H
    factor %= MOD
    factor = (factor + 1) % MOD

    m_sq = (M % MOD) * (M % MOD) % MOD
    ans = (factor * m_sq) % MOD

    print(ans)