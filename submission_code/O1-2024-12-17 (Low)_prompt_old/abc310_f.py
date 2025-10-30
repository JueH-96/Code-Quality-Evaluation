def solve():
    import sys
    sys.setrecursionlimit(10**7)

    MOD = 998244353
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # We want the probability that, after rolling die i in [1..A_i],
    # there is a subset of the N results that sums to exactly 10.
    #
    # Let total = product(A_i). That is the total number of equally-likely outcomes.
    #
    # We will count how many outcomes do NOT have any subset summing to 10.
    # Then #favorable = total - that count.
    #
    # Finally, we need (#favorable / total) % MOD, i.e. the modular fraction.
    # We compute favorable * inv(total) mod 998244353.

    # Approach:
    # We'll track, via bitmask dp, which sums up to 10 can be formed as a subset
    # of the chosen results so far. For the i-th die (which can be 1..A_i),
    # any value > 10 cannot help produce a sum up to 10. Hence for sums ≤ 10,
    # those large values act like "no new sum up to 10". But any value r in [1..10]
    # can shift existing sums by r. Because we are free to include or exclude
    # that rolled value in some subset.
    #
    # However, here's the key difference from typical subset-sum:
    # The outcome is fixed (the i-th die is exactly one of [1..A_i]), but
    # the subset can skip or use it. Either way, for sums up to 10, having
    # that i-th die able to be 1..min(A_i,10) means we can add new possible sums
    # from 1..min(A_i,10). And picking a value > 10 doesn't create sums ≤ 10.
    #
    # So effectively, for each partial "mask of sums" from the first i dice,
    # the set of sums from the first i+1 dice is:
    #    all old sums (bitmask) OR all old sums shifted by r in [1..min(A_i,10]].
    # All of those ways arise from any choice of the i-th die result in [1..A_i].
    #
    # Because from the perspective of "which sums are possible up to 10?",
    # picking any single r in [1..A_i] leads to the union of old sums and (old sums + r) capped at 10.
    #
    # And crucially, each old_mask transitions to exactly one new_mask
    # (the union of old_mask shifted by each r in [1..min(A_i, 10]] plus old_mask itself).
    #
    # The number of ways to pick the i-th die is A_i, so we multiply dp[i-1][old_mask]
    # by A_i for newdp[new_mask].
    #
    # In the end, dp[N][mask] is how many total outcomes produce exactly that set of possible sums up to 10.
    # We sum dp[N][mask] for all mask that do NOT contain sum=10 (i.e., the 10th bit is 0),
    # giving the count of outcomes for which no subset can sum to 10.
    #
    # Then #favorable = total - that count.
    # Probability = #favorable / total, output in mod 998244353.

    # Precompute the product of A_i (mod MOD).
    total = 1
    for val in A:
        total = (total * val) % MOD

    # dp array of length 2^11 (bit 0..10). 
    # dp[mask] = how many sequences of dice results so far produce exactly the set of sums indicated by mask.
    # Initially, with 0 dice, the only sum possible is {0}, i.e. bit 0 is set => mask = 1 << 0 = 1.
    dp = [0]*(1<<11)
    dp[1] = 1  # the set {0} only

    FULL_MASK = (1<<11) - 1

    for a in A:
        newdp = [0]*(1<<11)
        # For each old_mask, combine with all possible new values from the i-th die
        for old_mask in range(1<<11):
            ways = dp[old_mask]
            if ways == 0:
                continue
            # The new possible sums mask is the union of old_mask plus old_mask<<r for r in [1..min(a,10]]
            new_mask = old_mask
            limit = min(a, 10)
            for r in range(1, limit+1):
                shifted = (old_mask << r) & FULL_MASK
                new_mask |= shifted

            # All ways from dp[old_mask] expand to new_mask with factor = a
            newdp[new_mask] = (newdp[new_mask] + ways*a) % MOD

        dp = newdp

    # Count how many outcomes do NOT have sum=10 possible => bit 10 is off
    no_sum_10 = 0
    BIT_10 = 1 << 10
    for mask in range(1<<11):
        if (mask & BIT_10) == 0:  # bit 10 not set
            no_sum_10 = (no_sum_10 + dp[mask]) % MOD

    favorable = (total - no_sum_10) % MOD
    # Probability = favorable / total mod => favorable * inv(total) mod
    inv_total = pow(total, MOD-2, MOD)
    ans = (favorable * inv_total) % MOD

    print(ans)