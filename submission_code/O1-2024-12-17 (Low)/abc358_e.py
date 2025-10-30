def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    K = int(input_data[0])
    C = list(map(int, input_data[1:]))

    MOD = 998244353

    # We only need to handle sequence lengths up to K (≤ 1000).
    # We will use a dynamic programming approach where:
    # dp[i][j] = number of distinct strings of length j using only the
    # first i letters (A..(A+i-1)) such that the count of each letter
    # does not exceed its limit.
    # Finally, dp[26][1] + dp[26][2] + ... + dp[26][K] gives the answer.

    # However, storing dp in 2D would be large but we only need dp for the previous i
    # at each step. So we can do a 1D rolling array:
    # dp[j] after processing i-th letter.

    # Recurrence:
    # dp_new[j] = sum_{k=0..min(C_i, j)} [ comb(j, k) * dp_old[j-k] ]
    # Explanation:
    #  - We choose how many times (k) we use the i-th letter in a length-j string
    #  - We can place those k occurrences among the j positions in comb(j,k) ways
    #  - The remaining j-k positions are filled by strings counted in dp_old[j-k]
    #    (which uses i-1 letters with constraints)

    # Precompute factorials and inverse factorials up to 1000 for comb(n,k).
    maxN = K  # up to 1000
    fact = [1]*(maxN+1)
    invfact = [1]*(maxN+1)
    for i in range(1, maxN+1):
        fact[i] = fact[i-1]*i % MOD

    # Fermat's little theorem for inverse factorial
    invfact[maxN] = pow(fact[maxN], MOD-2, MOD)
    for i in reversed(range(maxN)):
        invfact[i] = invfact[i+1]*(i+1) % MOD

    def comb(n, r):
        if r<0 or r>n: return 0
        return fact[n]*invfact[r]%MOD*invfact[n-r]%MOD

    dp = [0]*(K+1)
    dp[0] = 1

    for limit in C:
        new_dp = [0]*(K+1)
        for length in range(K+1):
            if dp[length] == 0:
                # If dp[length] is 0, it won't contribute to new length > length in sum,
                # but we still can place 0 occurrences of this new letter to remain length.
                # So we do the k=0 term manually (it is comb(length,0)*dp[length] = dp[length]).
                # We skip a loop for k=0..0? We'll handle it directly or incorporate in the loop below.
                new_dp[length] = (new_dp[length] + dp[length]) % MOD
                continue

            val_old = dp[length]  # dp[i-1][length]
            # If we place none of the new letter in a length "length" string:
            new_dp[length] = (new_dp[length] + val_old) % MOD

        # Now handle the possibility of increasing length by adding the new letter (k>0).
        # Actually, we must loop over new length j from current length+1 .. up to K
        # and see if we can place some k of the new letter.
        # But the original formula is easier to interpret as:
        # new_dp[j] += sum_{k=0..min(limit,j)} comb(j, k)*dp_old[j-k].
        # We'll do exactly that approach for each j.
        # We already accounted for k=0 above but let's unify all in one snippet
        # (the above approach was partially done, but let's just do the standard formula).
        new_dp = [0]*(K+1)  # re-init to do the standard formula cleanly
        for j in range(K+1):
            # sum_{k=0..min(limit,j)} comb(j,k)*dp[j-k]
            # We'll accumulate in new_dp[j].
            s = 0
            # k can go up to min(limit, j).
            # dp[j-k] is the old dp for length j-k.
            # comb(j,k) chooses positions for the new letter among j positions.
            # This is the direct formula:
            # dp[i][j] = sum_{k=0..min(limit,j)} comb(j, k)*dp[i-1][j-k].
            #
            # We'll do it in a loop.
            #
            # NOTE: We can optimize with partial prefix sums if needed,
            # but let's first do the direct version; it might still pass.
            max_k = min(limit, j)
            acc = 0
            for k in range(max_k+1):
                acc += comb(j, k)*dp[j-k]
            new_dp[j] = acc % MOD
        dp = new_dp

    ans = sum(dp[1:K+1]) % MOD
    print(ans)