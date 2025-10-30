import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    mod = 998244353
    # Precompute C_l and Sx_l for lengths 1..6
    C = [0] * 7
    Sx = [0] * 7
    # For length l, numbers from lo to hi inclusive
    pow10 = [1] * 8
    for i in range(1, 8):
        pow10[i] = pow10[i-1] * 10
    for l in range(1, 7):
        lo = pow10[l-1]
        hi = min(pow10[l] - 1, N)
        if lo > hi:
            C[l] = 0
            Sx[l] = 0
        else:
            cnt = hi - lo + 1
            C[l] = cnt
            # sum of arithmetic sequence from lo to hi
            Sx[l] = (lo + hi) * cnt // 2 % mod
    # Precompute factorials and inv factorials up to N
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % mod
    invf = [1] * (N + 1)
    invf[N] = pow(fact[N], mod-2, mod)
    for i in range(N, 0, -1):
        invf[i-1] = invf[i] * i % mod

    # NTT / convolution setup
    root = 3
    rev_cache = {}
    root_cache = {}
    def get_rev(n):
        # bit-reversed indices for length n (power of 2)
        if n in rev_cache:
            return rev_cache[n]
        bits = n.bit_length() - 1
        rev = [0] * n
        for i in range(1, n):
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
        rev_cache[n] = rev
        return rev
    def get_root_pows(n):
        # precompute wlen and wlen_inv for stages for length n
        if n in root_cache:
            return root_cache[n]
        logn = n.bit_length() - 1
        wlen = [0] * logn
        wlen_inv = [0] * logn
        # stage i: block size = 2^(i+1)
        for i in range(logn):
            length = 1 << (i+1)
            # primitive root^( (mod-1) / length )
            w = pow(root, (mod-1) // length, mod)
            wlen[i] = w
            # inverse of w
            wlen_inv[i] = pow(w, mod-2, mod)
        root_cache[n] = (wlen, wlen_inv)
        return root_cache[n]

    def ntt(a, invert):
        n = len(a)
        rev = get_rev(n)
        # bit-reverse permutation
        for i in range(n):
            j = rev[i]
            if i < j:
                a[i], a[j] = a[j], a[i]
        wlen, wlen_inv = get_root_pows(n)
        logn = len(wlen)
        for stage in range(logn):
            length = 1 << (stage + 1)
            half = length >> 1
            wcur = wlen_inv[stage] if invert else wlen[stage]
            for i in range(0, n, length):
                w = 1
                base = i
                # unroll inner
                for j in range(half):
                    u = a[base + j]
                    v = a[base + j + half] * w % mod
                    # a[base + j] = u + v mod
                    if u + v < mod:
                        a[base + j] = u + v
                    else:
                        a[base + j] = u + v - mod
                    # a[base + j + half] = u - v mod
                    if u >= v:
                        a[base + j + half] = u - v
                    else:
                        a[base + j + half] = u - v + mod
                    w = w * wcur % mod
        if invert:
            inv_n = pow(n, mod-2, mod)
            for i in range(n):
                a[i] = a[i] * inv_n % mod

    def convolution(a, b):
        na = len(a)
        nb = len(b)
        if na == 0 or nb == 0:
            return []
        res_len = na + nb - 1
        # find power-of-two length
        n = 1
        while n < res_len:
            n <<= 1
        fa = a[:] + [0] * (n - na)
        fb = b[:] + [0] * (n - nb)
        ntt(fa, False)
        ntt(fb, False)
        for i in range(n):
            fa[i] = fa[i] * fb[i] % mod
        ntt(fa, True)
        # truncate to needed length
        del fa[res_len:]
        return fa

    # Build the total generating polynomial P_total(u) = ∏_{l=1..6} (1 + w_l * u)^{C[l]}
    poly = [1]
    # naive threshold
    NAIVE_THRESHOLD = 5_000_000
    for l in range(1, 7):
        cnt = C[l]
        if cnt == 0:
            continue
        # weight for length l
        wl = pow(10, l, mod)
        # build factor poly_f of length cnt+1: coeff[k] = C(cnt, k) * wl^k
        poly_f = [0] * (cnt + 1)
        # precompute binomial(n, k) via fact, invf
        f_cnt = fact[cnt]
        pw = 1
        for k in range(cnt + 1):
            # C(cnt, k) = fact[cnt] * invf[k] * invf[cnt-k]
            poly_f[k] = f_cnt * invf[k] % mod * invf[cnt - k] % mod * pw % mod
            pw = pw * wl % mod
        # multiply poly by poly_f
        na = len(poly)
        nb = cnt + 1
        if na * nb <= NAIVE_THRESHOLD:
            # naive multiply
            res = [0] * (na + nb - 1)
            for i in range(na):
                ai = poly[i]
                if ai:
                    rij = res  # local ref
                    bf = poly_f
                    for j in range(nb):
                        rij[i+j] = (rij[i+j] + ai * bf[j]) % mod
            poly = res
        else:
            # NTT convolution
            poly = convolution(poly, poly_f)
        # truncate to at most N+1 (we only need degrees up to N)
        if len(poly) > N+1:
            del poly[N+1:]

    # Ensure poly has length at least N
    if len(poly) < N:
        poly += [0] * (N + 1 - len(poly))

    # Compute T[K] = poly[K] * K! % mod for K = 0..N-1
    T = [0] * N
    for K in range(N):
        T[K] = poly[K] * fact[K] % mod

    # For each length l, compute W[l] via recurrence and sum
    ans = 0
    # Pre-get factorials for (N-1-K)! from fact
    # fact[t] = t! so (N-1-K)! = fact[N-1-K]
    for l in range(1, 7):
        if C[l] == 0:
            continue
        wl = pow(10, l, mod)
        # M[0] = 1
        Mprev = 1
        # sum over K=0..N-1 of M[K] * (N-1-K)!
        total = Mprev * fact[N-1] % mod
        # compute for K=1..N-1
        # M[K] = T[K] - K * wl * M[K-1]
        # accumulate total
        mul_wl = wl  # for use
        for K in range(1, N):
            # T[K] - K * wl * Mprev
            val = T[K] - (K * mul_wl) % mod * Mprev % mod
            if val < 0:
                val += mod
            Mprev = val
            total = (total + Mprev * fact[N-1-K]) % mod
        # W_l = total
        # add contribution Sx[l] * total
        ans = (ans + Sx[l] * total) % mod

    print(ans)

if __name__ == "__main__":
    main()