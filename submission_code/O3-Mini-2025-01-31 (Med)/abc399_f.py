def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    mod = 998244353

    # Precompute binomials C(K, j) for j = 0..K
    # Since K is small, we can compute cumulatively.
    binom = [0] * (K + 1)
    binom[0] = 1
    for j in range(1, K + 1):
        # binom[j] = binom[j-1] * (K - j + 1) / j  modulo mod.
        binom[j] = binom[j - 1] * (K - j + 1) % mod
        invj = pow(j, mod - 2, mod)
        binom[j] = (binom[j] * invj) % mod

    # Precompute sign factors: sign[x] = (-1)^x mod mod for x = 0..K.
    sign = [1] * (K + 1)
    for j in range(1, K + 1):
        sign[j] = (-1 * sign[j - 1]) % mod

    # F[t] will store the cumulative sum over previous prefix sums raised to power t.
    # Initially, we have prefix[0] = 0. By convention, 0^0 = 1.
    # For t >= 1, 0^t = 0.
    F = [0] * (K + 1)
    F[0] = 1  # because every term ^0 is taken as 1.

    prefix = 0
    ans = 0
    # We iterate over indices 1..n (processing each new prefix sum)
    for i in range(n):
        prefix = (prefix + A[i]) % mod
        # Compute powers[0..K] where powers[j] = prefix^j mod mod.
        powers = [1] * (K + 1)
        for j in range(1, K + 1):
            powers[j] = powers[j - 1] * prefix % mod

        # The key observation is that any subarray sum from x+1 to i (where x < i)
        # equals prefix(i) - prefix(x). We want to add up (prefix(i) - prefix(x))^K for all valid x.
        # Using the binomial expansion:
        #   (prefix(i) - prefix(x))^K = sum_{j=0}^K (-1)^(K-j)*C(K, j) * (prefix(i))^j * (prefix(x))^(K-j)
        # For the current index i, summing over all previous x (which are accumulated in F)
        # we get:
        #
        #   contribution_i = sum_{j=0}^K (-1)^(K-j)*C(K, j) * (prefix(i))^j * F[K-j]
        for j in range(K + 1):
            cur = sign[K - j] * binom[j] % mod
            cur = cur * powers[j] % mod
            cur = cur * F[K - j] % mod
            ans = (ans + cur) % mod

        # Update cumulative sums F[t] with the current prefix power
        for t in range(K + 1):
            F[t] = (F[t] + powers[t]) % mod

    sys.stdout.write(str(ans % mod))


if __name__ == '__main__':
    main()