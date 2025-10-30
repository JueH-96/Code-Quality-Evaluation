import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    M = 998244353

    # Precompute factorials and inverse factorials up to K
    fact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % M
    invfact = [1] * (K+1)
    invfact[K] = pow(fact[K], M-2, M)
    for i in range(K, 0, -1):
        invfact[i-1] = invfact[i] * i % M

    # Precompute coefficients: coef[t] = (-1)^(K-t) * C(K, t) mod M
    coef = [0] * (K+1)
    for t in range(K+1):
        c = fact[K] * invfact[t] % M * invfact[K-t] % M
        if ((K - t) & 1) == 1:
            coef[t] = (M - c) % M
        else:
            coef[t] = c

    # sum_u[u] will hold sum_{i < j} S_i^u (at the j-th iteration,
    # it holds sum_{i=0..j-1} S_i^u).  We initialize with S_0 = 0:
    # S_0^0 = 1, and S_0^u = 0 for u >= 1.
    sum_u = [0] * (K+1)
    sum_u[0] = 1

    # f[t] will accumulate sum_{1 <= j <= N} S_j^t * (sum_{i<j} S_i^{K-t})
    f = [0] * (K+1)

    pw = [0] * (K+1)  # temporary array for powers of current prefix sum
    S = 0

    for _ in range(N):
        a = int(next(it))
        S = (S + a) % M

        # compute pw[p] = S^p mod M for p = 0..K
        pw[0] = 1
        for p in range(1, K+1):
            pw[p] = pw[p-1] * S % M

        # update f[t] using old sum_u[K-t]
        # then update sum_u[u] by adding pw[u]
        for t in range(K+1):
            # f[t] += pw[t] * sum_u[K-t]
            f[t] = (f[t] + pw[t] * sum_u[K-t]) % M
        for u in range(K+1):
            sum_u[u] = (sum_u[u] + pw[u]) % M

    # combine with binomial coefficients and signs
    ans = 0
    for t in range(K+1):
        ans = (ans + coef[t] * f[t]) % M

    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()