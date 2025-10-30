def main():
    import sys
    # Use sys.stdin.buffer for fast I/O.
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # Compute prefix sums P[0..N] (with P[0]=0)
    P = [0]*(N+1)
    for i in range(N):
        P[i+1] = (P[i] + A[i]) % mod

    # Precompute F[e][r] = (P[r])^e mod mod for e=0,...,K and r=0,...,N.
    # It is more efficient to loop over r and compute all powers of P[r] at once.
    F = [[0]*(N+1) for _ in range(K+1)]
    for r in range(N+1):
        x = P[r]
        pwr = 1
        for e in range(K+1):
            F[e][r] = pwr
            pwr = (pwr * x) % mod

    # Precompute cumulative sums: cup[e][r] = Σ (from i=0 to r) F[e][i] mod mod.
    cup = [[0]*(N+1) for _ in range(K+1)]
    for e in range(K+1):
        s = 0
        Fe = F[e]
        Ce = cup[e]
        for r in range(N+1):
            s += Fe[r]
            if s >= mod:
                s -= mod
            Ce[r] = s

    # Precompute binomial coefficients C(K, j) for j=0..K.
    # Since K is very small (K<=10), we can use math.comb.
    try:
        from math import comb
        binom = [comb(K, j) % mod for j in range(K+1)]
    except ImportError:
        binom = [1]*(K+1)
        for j in range(K+1):
            num = 1
            den = 1
            for t in range(1, j+1):
                num *= (K - t + 1)
                den *= t
            binom[j] = (num // den) % mod

    # Precompute sign factors: for each j, we need factor = (-1)^(K-j) mod mod.
    # (Recall that in modulo arithmetic, -1 is represented as mod-1.)
    sign = [1]*(K+1)
    for j in range(K+1):
        # if (K-j) is odd then (-1)^(K-j) = -1 mod mod else 1.
        if ((K - j) & 1):
            sign[j] = mod - 1
        else:
            sign[j] = 1

    # Our formula (after reordering the summation) is:
    #   S_total = Σ_{j=0}^{K} C(K, j)·(-1)^(K-j) · ( Σ_{r=1}^{N} [ F[j][r] * cup[K-j][r-1] ] )
    #
    # Here:
    #   • F[j][r] = (P[r])^j, and
    #   • cup[K-j][r-1] = Σ_{l=0}^{r-1} (P[l])^(K-j).
    #
    # We now sum over j and r. We swap the order of summation so that the outer loop is over j.
    ans = 0
    # Loop over each fixed j. (j corresponds to the power on P[r])
    for j in range(K+1):
        coef = (binom[j] * sign[j]) % mod
        summ = 0
        Fj = F[j]             # List of (P[r])^j for r=0,...,N.
        C_other = cup[K - j]    # List corresponding to exponent (K-j).
        # We need to sum over r from 1 to N:
        #    Σ_{r=1}^{N} Fj[r] * C_other[r-1].
        # r runs from 1 to N, so we use Fj[1:] and C_other[0:N]
        for a, b in zip(Fj[1:], C_other[:N]):
            summ = (summ + a * b) % mod
        ans = (ans + coef * summ) % mod
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()