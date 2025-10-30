import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    mod = 998244353
    root = 3

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
            wlen = pow(root, (mod - 1) // length, mod)
            if invert:
                wlen = pow(wlen, mod-2, mod)
            for i in range(0, n, length):
                w = 1
                half = length >> 1
                for j in range(i, i + half):
                    u = a[j]
                    v = a[j + half] * w % mod
                    a[j] = u + v if u + v < mod else u + v - mod
                    a[j + half] = u - v if u >= v else u - v + mod
                    w = w * wlen % mod
            length <<= 1
        if invert:
            inv_n = pow(n, mod-2, mod)
            for i in range(n):
                a[i] = a[i] * inv_n % mod

    def poly_mul(a, b, need):
        # multiply polynomials a and b, return up to degree need-1
        la = len(a); lb = len(b)
        size = 1
        total = la + lb - 1
        while size < total:
            size <<= 1
        fa = a + [0] * (size - la)
        fb = b + [0] * (size - lb)
        ntt(fa, False); ntt(fb, False)
        for i in range(size):
            fa[i] = fa[i] * fb[i] % mod
        ntt(fa, True)
        if total > need:
            total = need
        return fa[:total]

    data = sys.stdin.read().split()
    K = int(data[0])
    C = list(map(int, data[1:]))

    # precompute factorials and inverses
    fact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % mod
    invfact = [1] * (K+1)
    invfact[K] = pow(fact[K], mod-2, mod)
    for i in range(K, 0, -1):
        invfact[i-1] = invfact[i] * i % mod

    # A(x) = current poly, A[l] = dp[l]/l!
    A = [1]  # degree 0

    for ci in C:
        # B(x) = sum_{t=0..ci} x^t / t!
        m = min(ci, K)
        B = invfact[:m+1]
        # Convolve A and B, truncate to degree K+1
        A = poly_mul(A, B, K+1)

    # compute answer: sum_{l=1..K} A[l] * l!
    ans = 0
    for l in range(1, len(A)):
        ans = (ans + A[l] * fact[l]) % mod
    print(ans)

if __name__ == "__main__":
    main()