def main():
    import sys
    import numpy as np
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    mod = 998244353

    # Read inputs
    K = int(input_data[0])
    C = list(map(int, input_data[1:]))

    # We need to count the number of strings of length L (1<=L<=K)
    # that are an arrangement (order matters) of letters, where for each letter,
    # its count is at most C[i]. If we fix the counts x_i (with 0<= x_i <= C[i])
    # and L = sum_i x_i, the number of strings with that frequency is
    # L! / (x_1! * x_2! * ... * x_26!).
    #
    # Notice that if we define a generating function
    #     G(z) = Product_{i=1}^{26} [sum_{x=0}^{C_i} z^x/(x!)]
    # then, writing
    #     G(z) = sum_{L=0}^K q[L] * z^L,
    # the number of valid strings with total length L is
    #     a[L] = L! * q[L].
    # So the desired answer is sum_{L=1}^K L! * q[L] modulo mod.
    #
    # We can compute q[0..K] (polynomial coefficients) by doing a convolution
    # product of the 26 small polynomials
    #     P_i(z) = sum_{x=0}^{min(C_i,K)} z^x/(x!)  mod mod.
    #
    # In our modular arithmetic we precompute factorials and inverse factorials.
    # Finally, we will compute each convolution using numpy's fast convolution method.

    # Precompute factorial and inverse factorial arrays up to K.
    fact = [1] * (K + 1)
    invfact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = fact[i - 1] * i % mod
    invfact[K] = pow(fact[K], mod - 2, mod)  # Fermat inverse of fact[K]
    for i in range(K, 0, -1):
        invfact[i - 1] = invfact[i] * i % mod

    # poly will hold the coefficients of Q(z)= sum_{L} poly[L] * z^L.
    # Start with poly = 1.
    poly = np.zeros(K + 1, dtype=np.int64)
    poly[0] = 1

    # For each letter, convolve with the polynomial P_i.
    for i in range(26):
        # The maximum power we care for is min(C[i], K)
        degree = min(C[i], K)
        # Build polynomial P_i with coefficients P_i[x] = 1/(x!) mod mod.
        Pi = np.empty(degree + 1, dtype=np.int64)
        for j in range(degree + 1):
            Pi[j] = invfact[j]
        # Use numpy.convolve to perform polynomial multiplication.
        # Truncate the result to degree K.
        conv = np.convolve(poly, Pi)[:K + 1] % mod
        poly = conv

    # Now poly[L] is q[L]. The answer is sum_{L=1}^{K} (fact[L] * poly[L]) mod mod.
    ans = 0
    for L in range(1, K + 1):
        ans = (ans + fact[L] * int(poly[L])) % mod

    sys.stdout.write(str(ans))


if __name__ == "__main__":
    main()