def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:] ))
    MOD = 998244353

    # We only care about subset sums up to 10, so we'll use an 11-bit mask (bits 0..10).
    # If bit s is set in a mask, it means "sum s is achievable by some subset."

    # Precompute the "shift" operation for each mask and each x in [1..10].
    # SHIFT[mask][x] = (mask) OR (mask << x), clipped to 11 bits.
    # This represents adding x to all subset-sums encoded in mask.
    SHIFT = [[0]*11 for _ in range(1<<11)]
    for mask in range(1<<11):
        for x in range(1, 11):
            SHIFT[mask][x] = mask | ((mask << x) & ((1 << 11) - 1))

    # dp[mask] will hold the number of ways (count of outcomes) such that
    # the sums of subsets formed by the dice so far is exactly the set indicated by 'mask'.
    dp = [0]*(1<<11)

    # Initially, with no dice, the only possible sum is 0,
    # so the mask with only bit 0 set (which is 1 in decimal) has 1 way.
    dp[1] = 1

    for i in range(N):
        newdp = [0]*(1<<11)
        # A[i] can be large, but only values up to 10 matter for forming sum <= 10.
        # Values > 10 won't help form sum=10, though they still count as outcomes.
        c_above_10 = max(0, A[i] - 10)

        for mask in range(1<<11):
            ways = dp[mask]
            if ways == 0:
                continue

            # If the die shows a value > 10 (which happens c_above_10 times),
            # no new sums <= 10 are formed, so we stay in the same mask.
            if c_above_10 > 0:
                newdp[mask] = (newdp[mask] + ways * c_above_10) % MOD

            # For each possible face value from 1..min(10, A[i]), we update the mask.
            up_to = min(A[i], 10)
            for xval in range(1, up_to + 1):
                # SHIFT[mask][xval] sets bits for old sums plus old sums+xval
                shifted_mask = SHIFT[mask][xval]
                newdp[shifted_mask] = (newdp[shifted_mask] + ways) % MOD

        dp = newdp

    # The total number of outcomes is the product of A[i], modulo MOD.
    total_outcomes = 1
    for val in A:
        total_outcomes = (total_outcomes * val) % MOD

    # We want the number of outcomes that can form sum=10 in some subset.
    # That corresponds to all masks that have bit 10 (which is 1<<10) set.
    bit_for_10 = 1 << 10
    favorable_count = 0
    for mask in range(1<<11):
        if mask & bit_for_10:
            favorable_count = (favorable_count + dp[mask]) % MOD

    # Probability = favorable_count / total_outcomes (mod 998244353).
    # Use Fermat's little theorem to invert total_outcomes modulo 998244353.
    inv_total = pow(total_outcomes, MOD - 2, MOD)
    answer = (favorable_count * inv_total) % MOD

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()