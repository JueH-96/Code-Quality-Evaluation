def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    mod = 998244353

    # Read input
    N, M = map(int, input_data[:2])
    constraints = input_data[2:]

    # ------------------------------------------------------------
    # Precompute binomial coefficients comb[n][k] up to n=500
    # Using Pascal's rule: comb[n][k] = comb[n-1][k] + comb[n-1][k-1], modded
    # ------------------------------------------------------------
    maxN = N  # We'll build up to N
    comb = [[0]*(maxN+1) for _ in range(maxN+1)]
    for i in range(maxN+1):
        comb[i][0] = 1
        for j in range(1, i+1):
            comb[i][j] = (comb[i-1][j] + comb[i-1][j-1]) % mod

    # ------------------------------------------------------------
    # Store "disallowed" information:
    # disallowed[l][r] = list of positions X
    # meaning "the maximum in subarray [l..r] must NOT be at X".
    # We'll use 1-based indexing for convenience.
    # ------------------------------------------------------------
    disallowed = [[[] for _ in range(N+1)] for _ in range(N+1)]

    idx = 0
    for _ in range(M):
        L_i = int(constraints[idx]); R_i = int(constraints[idx+1]); X_i = int(constraints[idx+2])
        idx += 3
        disallowed[L_i][R_i].append(X_i)

    # ------------------------------------------------------------
    # dp[l][r] = number of valid ways (mod) to arrange a permutation
    # on the subarray [l..r] so that none of the constraints are violated.
    #
    # For a Cartesian-tree viewpoint: the root of [l..r] cannot be
    # any X that appears in disallowed[l][r].
    #
    # Recurrence (typical for counting permutations by "root picking"):
    #   dp[l][r] = sum_{k=l..r, k not disallowed}
    #       ( dp[l][k-1] * dp[k+1][r] * comb(r-l, k-l) ) mod
    #
    # We'll implement via increasing subarray length.
    # Base case: dp[l][l-1] = 1 (empty subarray).
    # ------------------------------------------------------------

    # Initialize dp array
    # We'll make it (N+2)x(N+2) so dp[i][i-1] is in range
    dp = [[0]*(N+2) for _ in range(N+2)]

    # Base case: empty subarray => 1 way
    for i in range(1, N+2):
        dp[i][i-1] = 1

    # Compute dp in order of subarray length
    for length in range(1, N+1):             # length from 1 to N
        for l in range(1, N - length + 2):  # l goes so that r = l+length-1 <= N
            r = l + length - 1
            n = length  # subarray size

            # If everything in [l..r] is disallowed as root, dp=0
            if len(disallowed[l][r]) == n:
                dp[l][r] = 0
                continue

            # We'll build up the sum of dp[l][k-1]*dp[k+1][r]*comb(n-1, k-l)
            # We'll store partial results in B[j], where j = k-l
            # and then subtract for any disallowed positions.
            sumVal = 0
            B = [0]*n

            # Compute each term
            for j in range(n):
                k = l + j
                left_val = dp[l][k-1]
                right_val = dp[k+1][r]
                prod = (left_val * right_val) % mod
                # multiply by comb(n-1, j)
                b = (prod * comb[n-1][j]) % mod
                B[j] = b
                sumVal += b
                if sumVal >= mod:
                    sumVal -= mod

            # Subtract disallowed roots
            for x in disallowed[l][r]:
                j = x - l  # index in B
                b = B[j]
                sumVal -= b
                if sumVal < 0:
                    sumVal += mod

            dp[l][r] = sumVal % mod

    # The result for the entire permutation is dp[1][N]
    print(dp[1][N] % mod)

# Don't forget to call main()
if __name__ == "__main__":
    main()