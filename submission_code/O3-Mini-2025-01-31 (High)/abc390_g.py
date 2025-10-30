# YOUR CODE HERE
def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    try:
        N = int(data[0])
    except:
        return
    mod = 998244353

    # --- Helper: modular inverse
    def inv(x):
        return pow(x, mod-2, mod)
    
    # --- Group numbers 1..N by digit-length.
    # For d-digit numbers: if d==1 then lower bound=1, else lower=10^(d-1).
    groups = []  # each element: (d, count, group_sum)
    d = 1
    while True:
        if d == 1:
            low = 1
        else:
            low = 10**(d-1)
        high = min(N, 10**d - 1)
        if low > N:
            break
        cnt = high - low + 1
        # Compute group sum = (low+high)*cnt//2  mod mod.
        group_sum = ((low + high) % mod) * (cnt % mod) % mod
        group_sum = group_sum * inv(2) % mod
        groups.append((d, cnt, group_sum))
        d += 1

    # --- For each group, build two polynomials P_F(t) and P_G(t) as lists.
    # P_F: length = cnt+1, where for r=0,..., cnt, coefficient = C(cnt, r) * (10^d)^r
    # P_G: length = cnt+1, with P_G[0] = 0 and for r>=1, coefficient = group_sum * C(cnt-1, r-1) * (10^d)^r.
    group_polys = []
    for (d, cnt, gsum) in groups:
        A = pow(10, d, mod)
        polyF = [0]*(cnt+1)
        polyF[0] = 1
        for r in range(1, cnt+1):
            polyF[r] = polyF[r-1] * (cnt - r + 1) % mod
            polyF[r] = polyF[r] * inv(r) % mod
            polyF[r] = polyF[r] * A % mod
        # Using the relation C(cnt-1, r-1) = C(cnt, r) * r/ cnt, we set:
        polyG = [0]*(cnt+1)
        polyG[0] = 0
        if cnt > 0:
            inv_cnt = inv(cnt)
            for r in range(1, cnt+1):
                polyG[r] = gsum * polyF[r] % mod
                polyG[r] = polyG[r] * r % mod
                polyG[r] = polyG[r] * inv_cnt % mod
        group_polys.append((polyF, polyG))
    
    # --- We now want to multiply all the group polynomials (“convolve” them).
    # The overall polynomials will be:
    #   Combined_F[j] = sum_{J ⊆ S, |J|=j} ∏ (10^(L(y)))  and
    #   Combined_G[j] = sum_{J ⊆ S, |J|=j} ( (sum_{x in J} x) ⋅ ∏ 10^(L(y)) ).
    # We will combine using FFT- based convolution.
    try:
        import numpy as np
    except ImportError:
        # If numpy not available, a naive convolution may be used (it will be too slow for large N)
        def conv_naive(a, b):
            n = len(a)
            m = len(b)
            res = [0]*(n+m-1)
            for i in range(n):
                for j in range(m):
                    res[i+j] = (res[i+j] + a[i]*b[j]) % mod
            return res
        def poly_conv(a, b):
            return conv_naive(a, b)
    else:
        def poly_conv(a, b):
            # a and b are 1D numpy arrays (dtype=int64)
            la = a.shape[0]
            lb = b.shape[0]
            L = la + lb - 1
            size = 1
            while size < L:
                size *= 2
            A_arr = np.array(a, dtype=np.int64)
            B_arr = np.array(b, dtype=np.int64)
            A_pad = np.zeros(size, dtype=np.float64)
            B_pad = np.zeros(size, dtype=np.float64)
            A_pad[:la] = A_arr
            B_pad[:lb] = B_arr
            FA = np.fft.rfft(A_pad)
            FB = np.fft.rfft(B_pad)
            FC = FA * FB
            c_pad = np.fft.irfft(FC, n=size)[:L]
            c_pad = np.rint(c_pad).astype(np.int64) % mod
            return c_pad

    # Define how to combine two pairs of polynomials.
    def combine(poly_pair1, poly_pair2):
        F1, G1 = poly_pair1
        F2, G2 = poly_pair2
        import numpy as np
        F1_arr = np.array(F1, dtype=np.int64)
        G1_arr = np.array(G1, dtype=np.int64)
        F2_arr = np.array(F2, dtype=np.int64)
        G2_arr = np.array(G2, dtype=np.int64)
        newF = poly_conv(F1_arr, F2_arr) % mod
        newG = (poly_conv(F1_arr, G2_arr) + poly_conv(G1_arr, F2_arr)) % mod
        return (newF.tolist(), newG.tolist())
    
    # Combine all group–polynomials (they are already sorted by digit–length)
    curF = [1]  # identity polynomial
    curG = [0]
    pair = (curF, curG)
    for poly_pair in group_polys:
        pair = combine(pair, poly_pair)
    combined_F, combined_G = pair
    total_count = sum(cnt for (_, cnt, _) in groups)
    # We should have total_count == N.
    
    # --- Final step.
    # One may show that (after some algebra) the answer is
    #   answer = (N–1)! * Σ₍j=0₎^(N–1) ((sumAll * Combined_F[j] – Combined_G[j]) / C(N–1,j)) mod mod.
    # But using
    #   inv(C(N–1,j)) = fact[j] * fact[N–1-j] * inv(fact[N–1]),
    # multiplying by (N–1)! cancels the inv(fact[N–1]) so that:
    #   answer = Σ₍j=0₎^(N–1) [ (sumAll*Combined_F[j] – Combined_G[j]) * fact[j] * fact[N–1-j] ] mod mod.
    sumAll = N * (N+1) // 2 % mod
    # Precompute factorials up to N.
    fact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1]*i % mod
    invfact = [1]*(N+1)
    invfact[N] = pow(fact[N], mod-2, mod)
    for i in range(N, 0, -1):
        invfact[i-1] = invfact[i]*i % mod

    tot = 0
    # our combined polynomials have length N+1. We sum j = 0 to N-1.
    for j in range(N):
        term = (sumAll * combined_F[j] - combined_G[j]) % mod
        term = term * (fact[j] * fact[N-1-j] % mod) % mod
        tot = (tot + term) % mod
    ans = tot % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()