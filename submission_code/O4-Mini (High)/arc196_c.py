import sys
def main():
    input = sys.stdin.readline
    MOD = 998244353
    root = 3

    # ----- Fast NTT / convolution -----
    def ntt(a, invert):
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        length = 2
        while length <= n:
            wlen = pow(root, (MOD - 1) // length, MOD)
            if invert:
                wlen = pow(wlen, MOD - 2, MOD)
            for i in range(0, n, length):
                w = 1
                half = length >> 1
                for j in range(i, i + half):
                    u = a[j]
                    v = a[j + half] * w % MOD
                    a[j] = u + v if u + v < MOD else u + v - MOD
                    a[j + half] = u - v if u >= v else u - v + MOD
                    w = w * wlen % MOD
            length <<= 1
        if invert:
            inv_n = pow(n, MOD - 2, MOD)
            for i in range(n):
                a[i] = a[i] * inv_n % MOD

    def convolution(a, b):
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return []
        n = 1
        while n < na + nb - 1:
            n <<= 1
        fa = a[:] + [0] * (n - na)
        fb = b[:] + [0] * (n - nb)
        ntt(fa, False)
        ntt(fb, False)
        for i in range(n):
            fa[i] = fa[i] * fb[i] % MOD
        ntt(fa, True)
        return fa[:na + nb - 1]

    # ----- polynomial inverse by Newton iteration -----
    def poly_inv(a, m):
        # want h so that a*h ≡ 1 mod x^m, a[0]=1
        h = [1]  # h mod x^1
        length = 1
        while length < m:
            length2 = min(2 * length, m)
            # compute h * a mod x^{length2}, then h <- h*(2 - a*h) mod x^{length2}
            s1 = convolution(h, a[:length2])
            s1 = s1[:length2]
            for i in range(length2):
                s1[i] = (-s1[i]) % MOD
            s1[0] = (s1[0] + 2) % MOD
            h = convolution(h, s1)
            h = h[:length2]
            length = length2
        return h

    # ---- read input ----
    N = int(input())
    S = input().strip()
    # must have first = 'B' and last = 'W'
    if S[0] != 'B' or S[-1] != 'W':
        print(0)
        return

    # collect the "balanced‐prefix" levels d_i = #blacks in prefix where #B = #W
    bcnt = wcnt = 0
    dlist = []
    for i in range(2 * N - 1):
        if S[i] == 'B':
            bcnt += 1
        else:
            wcnt += 1
        if bcnt == wcnt:
            dlist.append(bcnt)
    # dlist is strictly increasing and all <= N−1

    # ----- factorials up to N -----
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD

    # build G(x) = sum_{t=0..N} t! x^t
    G = fact[:]  # G[t] = t!

    # compute H(x) = 1/G(x)  mod x^(N+1)
    H = poly_inv(G, N + 1)
    # now H[t] = coefficient of x^t in 1/G(x) = (-1 sum_{i<t} H[i]*fact[t-i])

    # final answer = sum_{d in {0}+dlist} H[d] * fact[N - d]
    ans = H[0] * fact[N] % MOD
    for d in dlist:
        ans = (ans + H[d] * fact[N - d]) % MOD

    print(ans)


if __name__ == "__main__":
    main()