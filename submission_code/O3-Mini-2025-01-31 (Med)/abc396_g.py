def main():
    import sys
    import numpy as np
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    N = 1 << W  # 2^W

    # Build frequency array f for all rows (each row is an integer between 0 and 2^W-1)
    f = np.zeros(N, dtype=np.int64)
    for _ in range(H):
        s = next(it).strip()
        # Interpret the string s as a binary number; s[0] is the leftmost bit.
        r = int(s, 2)
        f[r] += 1

    # Build array g where for each bitmask s, g[s] = min(popcount(s), W - popcount(s))
    g = np.empty(N, dtype=np.int64)
    for x in range(N):
        cnt = x.bit_count()
        if cnt > W - cnt:
            cnt = W - cnt
        g[x] = cnt

    # Fast Walsh–Hadamard Transform (FWHT) for XOR convolution.
    def fwht(a, inverse=False):
        n = a.shape[0]
        step = 1
        while step < n:
            # Reshape the array so that operations are performed on blocks of size 2*step.
            a = a.reshape(-1, 2 * step)
            # Copy left and right half of each block.
            left = a[:, :step].copy()
            right = a[:, step: 2 * step].copy()
            a[:, :step] = left + right
            a[:, step: 2 * step] = left - right
            a = a.reshape(n)
            step *= 2
        if inverse:
            # Inverse transform: divide by n.
            a //= n
        return a

    # Compute FWHT on f and g.
    F = fwht(f.copy(), inverse=False)
    G = fwht(g.copy(), inverse=False)
    
    # Pointwise multiplication in the FWHT domain
    prod = F * G
    
    # Inverse FWHT to obtain the XOR convolution h[m] = sum_r f[r] * g(r XOR m)
    conv = fwht(prod, inverse=True)
    
    # The answer is the minimum value of the resulting convolution array.
    ans = int(conv.min())
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()