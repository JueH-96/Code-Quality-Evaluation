def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    MOD = 998244353
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = int(input_data[1])
    T = list(map(int, input_data[2:]))

    # ----------------------------------------------------------------
    # We want the probability that song #1 (length T[0]) is playing at time (X + 0.5).
    #
    # Let t_star = X + 0.5.
    # A random-play sequence is an i.i.d. choice of songs from {1..N}, each chosen w.p. 1/N,
    # played back-to-back. Denote the lengths by T_i. We want the event that t_star
    # lies inside the block of song #1.
    #
    # In other words, if S_k = T_{I_1} + ... + T_{I_k} is the time the k-th song ends,
    # then we want:
    #   S_{k-1} <= t_star < S_{k-1} + T_{I_k},   and  I_k = 1.
    #
    # Because T_i are integers and t_star is always .5 above an integer X, the condition
    # S_{k-1} <= X+0.5 < S_{k-1} + T_1  (since I_k=1 => length T_1 = T[0]) becomes:
    #   let s = S_{k-1},  s an integer.
    #   1) s <= X          (since s <= X+0.5 and s is integer)
    #   2) s + T_1 > X+0.5 => T_1 > X+0.5 - s => T_1 >= (X - s) + 1
    #      => s >= X+1 - T_1.
    #
    # So s must be in the integer range [max(0, X+1 - T_1) .. X].
    #
    # We also need the probability that S_{k-1} = s for some k-1 >= 0, times the probability
    # the next chosen song is #1 (which is 1/N).
    #
    # Let p_m(s) = P(S_m = s), sum of m i.i.d. picks from {T_1,..., T_N} each w.p. 1/N.
    # Then P(S_m = s) is the coefficient of x^s in [ G(x) ]^m, where
    #   G(x) = (1/N)* ( x^(T_1) + x^(T_2) + ... + x^(T_N) ).
    #
    # Summing over all m >= 0, the coefficient of x^s in 1/(1 - G(x)) is
    #   sum_{m=0..∞} p_m(s).
    #
    # Hence define H(x) = 1/(1 - G(x)), then H_s = sum_{m=0..∞} p_m(s).
    #
    # The probability that time (X+0.5) is in a block of song #1 is
    #   sum_{s = max(0, X+1 - T_1)}^X [ H_s * (1/N) ]    (all operations in probability sense).
    #
    # We will do all arithmetic modulo 998244353, working as formal power series.
    #
    # Steps:
    # 1) Build the array g up to x^X, where g[k] = (count of T_i = k)/N mod 998244353 (k=1..X).
    #    (T_i > X do not contribute to partial sums <= X, so they can be ignored in G(x)
    #     for coefficient indices <= X.)
    # 2) Define f(x) = 1 - g(x).  Then f(0) = 1 - g(0) = 1 because none of the T_i are 0-length.
    # 3) Compute H(x) = 1/f(x) up to x^X (i.e. we only need coefficients up to x^X).
    # 4) Let M = max(0, X+1 - T[0]).  Sum H[M],...,H[X], multiply by (1/N) mod. That is the answer.
    #
    # We implement polynomial inversion via Newton's method, using NTT for fast convolution.
    # ----------------------------------------------------------------
    
    # Precompute inverse of N
    invN = pow(N, MOD-2, MOD)
    
    # Build polynomial g of length X+1 (indices 0..X).
    # g[k] = sum of (1/N) for each T_i==k.
    # T_i ranges from 1..10000, but we only store up to X because partial sums <= X can't come
    # from summands > X in that partial sum.
    g = [0]*(X+1)
    for length in T:
        if length <= X:
            g[length] = (g[length] + invN) % MOD
    
    # f(x) = 1 - g(x). We'll store up to x^X. f[0] = 1, f[k] = -g[k] for k>=1.
    # We'll invert f(x) up to length X+1.
    f = [0]*(X+1)
    f[0] = (1 - g[0]) % MOD  # g[0] should be 0 anyway, but just in case
    for i in range(1, X+1):
        # f[i] = -g[i] mod
        f[i] = (-g[i]) % MOD
    
    # We need to compute H(x) = 1/f(x), truncated to x^X.
    # Implement polynomial inversion via NTT + Newton iteration.

    # ---------------------- NTT/Convolution Routines ------------------

    # Primitive root for MOD=998244353 is 3; it is used for the NTT.
    # We implement an in-place iterative NTT.
    def ntt(a, invert=False):
        """In-place NTT (Number Theoretic Transform) of array a."""
        n = len(a)
        j = 0
        # Bit-reversal permutation
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]

        length = 2
        # root^( (MOD-1) / n ) is a principal n-th root of unity for n= length
        # For invert, we use the inverse root.  The constant is precomputed or we compute on the fly.
        while length <= n:
            # wlen = pow(3, (MOD-1)//length, MOD)  # forward
            # Instead of computing each time, we might do an iterative approach, but let's do direct pow.
            step = (MOD - 1) // length
            wlen = pow(3, step, MOD)
            if invert:
                wlen = pow(wlen, MOD-2, MOD)
            i = 0
            while i < n:
                w = 1
                half = length >> 1
                for k in range(i, i + half):
                    u = a[k]
                    v = (a[k + half] * w) % MOD
                    a[k] = (u + v) % MOD
                    a[k + half] = (u - v) % MOD
                    w = (w * wlen) % MOD
                i += length
            length <<= 1

        if invert:
            inv_n = pow(n, MOD-2, MOD)
            for i in range(n):
                a[i] = (a[i] * inv_n) % MOD

    def poly_mul(A, B):
        """Convolution (polynomial multiplication) of A, B in mod, returning full product."""
        nA = len(A)
        nB = len(B)
        if nA == 0 or nB == 0:
            return []
        size = 1
        while size < nA + nB - 1:
            size <<= 1
        fA = A + [0]*(size - nA)
        fB = B + [0]*(size - nB)

        ntt(fA, False)
        ntt(fB, False)
        for i in range(size):
            fA[i] = (fA[i] * fB[i]) % MOD
        ntt(fA, True)
        # trim trailing zeros if desired, but let's keep full product
        return fA

    def poly_mul_trunc(A, B, length_out):
        """Multiply polynomials A,B mod, then truncate to first length_out coefficients."""
        if not A or not B:
            return [0]*length_out
        # multiply
        res = poly_mul(A, B)
        if len(res) < length_out:
            res += [0]*(length_out - len(res))
        return res[:length_out]

    # ---------------------- Polynomial Inversion via Newton's Method ----------------------
    def poly_inv(f, n):
        """
        Compute the first n coefficients of 1/f(x) mod x^n,
        given f(0) != 0 in mod.  Uses Newton iteration.
        """
        # r(x) s.t. f(x)*r(x) = 1 mod x^n
        # f(0) must be invertible => f(0) != 0.
        r = [0]*n
        # base case: r(0) = 1/f(0)
        invf0 = pow(f[0], MOD-2, MOD)
        r[0] = invf0

        size = 1
        while size < n:
            size2 = size << 1
            # we want to compute up to length size2
            # step1: multiply f[:size2], r[:size2] => conv (truncate size2)
            # step2: r_new = r + r*(0 - conv), but standard formula is r*(2 - f*r).
            # i.e. r_{k+1} = r_k (2 - f*r_k) mod x^(2^k).
            # We'll do:
            #   conv = f*r (size2)
            #   tmp = 2 - conv
            #   newr = r * tmp
            # then truncate to size2
            A = f[:size2] + [0]*(size2 - len(f[:size2]))
            B = r[:size2] + [0]*(size2 - len(r[:size2]))
            conv = poly_mul_trunc(A, B, size2)

            # tmp = 2 - conv
            # careful with mod
            for i in range(size2):
                conv[i] = (-conv[i]) % MOD
            conv[0] = (conv[0] + 2) % MOD

            newr = poly_mul_trunc(B, conv, size2)
            # copy back
            lim = min(size2, n)
            for i in range(lim):
                r[i] = newr[i]
            size = size2
        return r[:n]

    # Compute H(x) = 1/f(x) up to x^X.
    H = poly_inv(f, X+1)

    # M = max(0, X+1 - T_1)
    T1 = T[0]
    M = X+1 - T1
    if M < 0:
        M = 0
    if M > X:
        # Then there's no s in [M..X], so probability is 0.
        print(0)
        return

    # sumCoeffs = sum_{s=M..X} H[s]  (mod)
    sumCoeffs = 0
    for s in range(M, X+1):
        sumCoeffs = (sumCoeffs + H[s]) % MOD

    # Probability = sumCoeffs * (1/N) in mod
    prob = (sumCoeffs * invN) % MOD

    print(prob)