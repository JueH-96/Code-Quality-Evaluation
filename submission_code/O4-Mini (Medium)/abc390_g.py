import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    MOD = 998244353
    # 1) Precompute factorials up to N and inv factorials
    nmax = N
    fac = [1] * (nmax+1)
    ifac = [1] * (nmax+1)
    for i in range(1, nmax+1):
        fac[i] = fac[i-1] * i % MOD
    inv = pow(fac[nmax], MOD-2, MOD)
    ifac[nmax] = inv
    for i in range(nmax, 0, -1):
        ifac[i-1] = ifac[i] * i % MOD

    # 2) Partition 1..N by digit-length
    groups = []  # list of (d, C_d, sum_of_labels_S_d)
    d = 1
    start = 1
    while start <= N:
        end = min(N, 10**d - 1)
        cnt = end - start + 1
        # sum of arithmetic seq start..end
        S = (start + end) * cnt // 2 % MOD
        groups.append((d, cnt, S))
        d += 1
        start *= 10

    # 3) NTT implementation
    def _ntt(a, invert):
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
        root = 3
        # precompute roots of unity
        length = 2
        while length <= n:
            wlen = pow(root, (MOD-1)//length, MOD)
            if invert:
                wlen = pow(wlen, MOD-2, MOD)
            for i in range(0, n, length):
                w = 1
                half = length>>1
                for j in range(i, i+half):
                    u = a[j]
                    v = a[j+half] * w % MOD
                    a[j] = u+v if u+v<MOD else u+v-MOD
                    a[j+half] = u-v if u>=v else u-v+MOD
                    w = w * wlen % MOD
            length <<= 1
        if invert:
            invn = pow(n, MOD-2, MOD)
            for i in range(n):
                a[i] = a[i] * invn % MOD

    def convolution(a, b, need):
        # multiply a by b, truncate to length=need
        la = len(a); lb = len(b)
        size = 1
        while size < la + lb - 1:
            size <<= 1
        fa = a + [0]*(size - la)
        fb = b + [0]*(size - lb)
        _ntt(fa, False)
        _ntt(fb, False)
        for i in range(size):
            fa[i] = fa[i] * fb[i] % MOD
        _ntt(fa, True)
        return fa[:need]

    # 4) Build dp = product over groups of (1 + a_d x)^{C_d}
    dp = [1]
    for (dig, cnt, S_d) in groups:
        # a_d = 10^dig mod
        a_d = pow(10, dig, MOD)
        # build polynomial f[t] = binom(cnt, t) * a_d^t  for t=0..cnt
        f = [0]*(cnt+1)
        # binom(cnt,t) = fac[cnt]*ifac[t]*ifac[cnt-t]
        base = fac[cnt]
        for t in range(cnt+1):
            f[t] = base * ifac[t] % MOD * ifac[cnt-t] % MOD * pow(a_d, t, MOD) % MOD
        # convolve dp by f, truncate to N+1
        new_len = min(len(dp) + len(f) - 1, N+1)
        dp = convolution(dp, f, new_len)

    # 5) precompute fac and reversed fac for (N-1-k)!: use fac_rev[k] = fac[N-1-k]
    fac_rev = [0]* (N+1)
    for k in range(N+1):
        if N-1-k >= 0:
            fac_rev[k] = fac[N-1-k]
        else:
            fac_rev[k] = 0

    # 6) For each group, compute P_d[k] = dp[k] - a_d * P_d[k-1]  (P_d[0]=dp[0])
    ans = 0
    for (dig, cnt, S_d) in groups:
        a_d = pow(10, dig, MOD)
        # build P array length N
        P = [0]*N
        P[0] = dp[0]
        for k in range(1, N):
            # only need up to N-1
            P[k] = (dp[k] - a_d * P[k-1]) % MOD
        # sum C_d = sum_{k=0..N-1} P[k] * k! * (N-1-k)!
        Cd = 0
        for k in range(N):
            Cd = (Cd + P[k] * fac[k] % MOD * fac_rev[k]) % MOD
        # add group contribution
        ans = (ans + S_d * Cd) % MOD

    print(ans)

if __name__ == "__main__":
    main()