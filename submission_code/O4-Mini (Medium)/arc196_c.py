import threading
import sys
def main():
    import sys
    sys.setrecursionlimit(1000000)
    mod = 998244353
    input = sys.stdin.readline

    N_line = input().strip()
    if not N_line:
        return
    N = int(N_line)
    S = input().strip()
    if S[0] != 'B' or S[-1] != 'W':
        print(0)
        return

    # Precompute factorials up to N
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % mod

    # Collect cut positions: where prefix #W == #B
    K = []  # K_i = number of blacks (== whites) in prefix
    cw = cb = 0
    for ch in S:
        if ch == 'W':
            cw += 1
        else:
            cb += 1
        if cw == cb:
            K.append(cb)
    # K[-1] == N guaranteed because S[0]=B,S[-1]=W
    m = len(K)
    # Svec holds the working dp RHS, dp will hold final values
    Svec = [fact[K[i]] for i in range(m)]
    dp = [0] * m

    # NTT implementation
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
        root = 3
        while length <= n:
            wlen = pow(root, (mod-1)//length, mod)
            if invert:
                wlen = pow(wlen, mod-2, mod)
            for i in range(0, n, length):
                w = 1
                half = length >> 1
                for j in range(i, i+half):
                    u = a[j]
                    v = a[j+half] * w % mod
                    a[j] = u + v if u + v < mod else u + v - mod
                    a[j+half] = u - v if u >= v else u - v + mod
                    w = w * wlen % mod
            length <<= 1
        if invert:
            inv_n = pow(n, mod-2, mod)
            for i in range(n):
                a[i] = a[i] * inv_n % mod

    def convolution(A, B):
        # returns A * B
        na = len(A)
        nb = len(B)
        n = 1
        while n < na + nb - 1:
            n <<= 1
        fa = A + [0] * (n - na)
        fb = B + [0] * (n - nb)
        ntt(fa, False)
        ntt(fb, False)
        for i in range(n):
            fa[i] = fa[i] * fb[i] % mod
        ntt(fa, True)
        return fa

    # CDQ divide and conquer
    def cdq(l, r):
        if l == r:
            # Finalize dp[l]
            dp[l] = Svec[l] % mod
            return
        mid = (l + r) >> 1
        cdq(l, mid)
        # Now propagate contributions from dp[l..mid] to Svec[mid+1..r]
        jl = l; jm = mid; jr = r
        sizeJ = jm - jl + 1
        sizeI = jr - jm
        # naive threshold: if small blocks, do direct
        if sizeJ * sizeI <= 2000000:
            for j in range(jl, jm+1):
                dj = dp[j]
                Kj = K[j]
                if dj:
                    for i in range(jm+1, jr+1):
                        d = K[i] - Kj
                        # d >= 0 by K increasing
                        Svec[i] = (Svec[i] - dj * fact[d]) % mod
        else:
            # use NTT
            K_l = K[jl]
            K_m = K[jm]
            K_r = K[jr]
            a_len = K_m - K_l + 1
            b_len = K_r - K_l + 1
            # build A, B
            A = [0] * a_len
            for j in range(jl, jm+1):
                A[K[j] - K_l] = dp[j]
            B = fact[:b_len]
            C = convolution(A, B)
            # C[t] = sum_{k} A[k] * B[t-k]
            for i in range(jm+1, jr+1):
                idx = K[i] - K_l
                if idx < len(C):
                    Svec[i] = (Svec[i] - C[idx]) % mod
        cdq(mid+1, r)

    cdq(0, m-1)
    # dp[m-1] is answer
    print(dp[m-1] % mod)

if __name__ == "__main__":
    main()