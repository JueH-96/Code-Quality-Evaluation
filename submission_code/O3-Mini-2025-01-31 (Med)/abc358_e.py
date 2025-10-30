def main():
    import sys, numpy as np
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    K = int(data[0])
    Cs = list(map(int, data[1:27]))

    # Precompute factorials and inverse factorials modulo mod up to K.
    fact = [1] * (K + 1)
    invfact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = fact[i - 1] * i % mod
    invfact[K] = pow(fact[K], mod - 2, mod)
    for i in range(K, 0, -1):
        invfact[i - 1] = invfact[i] * i % mod

    # dp will represent our generating function polynomial.
    # It is defined so that the coefficient at x^L, dp[L], equals:
    #    sum (over distributions for letters so far) of (1/(n1! n2! ... )) mod mod,
    # where L = n1+n2+... and each letter's count is within allowed limits.
    # Because a distribution with counts (n1,...,n26) yields L!/(n1! n2! ...),
    # we will later multiply dp[L] by fact[L] to get the number of strings.
    dp = np.zeros(K + 1, dtype=np.int64)
    dp[0] = 1

    # For each letter we multiply our generating function by the polynomial:
    #    poly_i(x) = sum_{j=0}^{min(C_i, K)} x^j / j!
    # We only need coefficients up to x^K.
    for i in range(26):
        max_use = Cs[i] if Cs[i] <= K else K
        poly = np.empty(max_use + 1, dtype=np.int64)
        # poly[j] = invfact[j] (which is the modular inverse of j!)
        poly[:] = invfact[:max_use + 1]
        # Convolve current dp with poly.
        # np.convolve returns the full polynomial; we take only coefficients up to degree K.
        dp = np.convolve(dp, poly)[:K + 1] % mod

    # Now, for each term of degree L (with 1 <= L <= K), the number of strings using
    # that distribution is dp[L] * fact[L] (since L!/(n1! n2! ...)* (n1! n2! ...) = L!).
    result = 0
    for L in range(1, K + 1):
        result = (result + dp[L] * fact[L]) % mod

    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()