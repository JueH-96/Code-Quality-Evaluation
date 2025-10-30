def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    mod = 998244353

    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Precompute the product of A_i modulo mod
    # This is the denominator of the probability.
    M = 1
    for x in A:
        M = (M * x) % mod

    # We will use a DP over the "set of possible subset sums" up to 10.
    # Represent the set of possible sums as a bitmask of length 11 (bits 0..10).
    # bit s = 1 means "sum s is achievable by some subset."
    # dp[mask] = how many outcomes (over dice processed so far) produce exactly that bitmask of possible sums.

    # Initially, with 0 dice, only sum=0 is possible, so bitmask = (1 << 0).
    dp = [0] * (1 << 11)
    dp[1 << 0] = 1  # only the empty subset sum = 0

    # Precompute how to combine masks:
    # - combmask[mask][k]: the new mask if we union the old sums plus old sums + k, clamped to 10.
    # - shiftGT10[mask]  : the new mask if we add any r >= 11, meaning any old sum < 10 forces sum=10.
    MAX_SUM = 10
    FULL_MASK = (1 << (MAX_SUM + 1)) - 1  # for bitwise clamping, here it's 0..10 inclusive => 11 bits

    combmask = [[0]*11 for _ in range(1 << 11)]  
    # combmask[mask][k] for k in 1..10

    for mask in range(1 << 11):
        for k in range(1, MAX_SUM+1):
            shifted = (mask << k) & FULL_MASK
            combmask[mask][k] = mask | shifted

    shiftGT10 = [0]*(1 << 11)
    for mask in range(1 << 11):
        # If there's any bit < 10 set, then adding >= 11 means we definitely can reach sum=10.
        if (mask & ((1 << MAX_SUM) - 1)) != 0:
            # set bit 10 as well
            shiftGT10[mask] = mask | (1 << MAX_SUM)
        else:
            shiftGT10[mask] = mask
            # if mask already has bit 10 set, it stays set anyway

    # Now run the DP over N dice
    idx = 0
    for i in range(N):
        dp_new = [0]*(1 << 11)
        a = A[i]
        for mask in range(1 << 11):
            count = dp[mask]
            if count == 0:
                continue
            # For each possible result r in [1..a] for this die,
            # we unify the old sums (skip) and old sums + r (pick).
            # But we don't literally loop r=1..a (which can be too large),
            # we note that r in 1..10 is distinct, and r>=11 all collapse the same way.
            up_to_10 = min(a, 10)
            for r in range(1, up_to_10+1):
                dp_new[combmask[mask][r]] = (dp_new[combmask[mask][r]] + count) % mod
            if a > 10:
                # This is the effect of all r in [11..a], i.e. (a-10) many times
                add_val = (count * (a - 10)) % mod
                dp_new[shiftGT10[mask]] = (dp_new[shiftGT10[mask]] + add_val) % mod
        dp = dp_new

    # Number of outcomes (among all possible dice results) that can form sum=10
    # is the sum of dp[mask] for those masks that have bit 10 set
    bit_10 = 1 << 10
    good_count = 0
    for mask in range(1 << 11):
        if (mask & bit_10) != 0:
            good_count = (good_count + dp[mask]) % mod

    # Probability = good_count / M  (mod 998244353)
    # i.e. good_count * M^{-1} mod 998244353
    # Compute modular inverse of M
    inv_M = pow(M, mod - 2, mod)
    ans = (good_count * inv_M) % mod

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()