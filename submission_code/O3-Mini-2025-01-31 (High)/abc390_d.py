def main():
    import sys
    sys.setrecursionlimit(10000)
    
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    total_masks = 1 << N
    # Precompute the sum for every subset (mask) of the N bags.
    subset_sum = [0] * total_masks
    # For each nonzero mask, use the trick: mask = (mask without its lowest set bit) plus that element
    for mask in range(1, total_masks):
        lowbit = mask & -mask  # gets the lowest set bit in mask
        idx = lowbit.bit_length() - 1  # 0-indexed bag number corresponding to lowbit
        subset_sum[mask] = subset_sum[mask - lowbit] + A[idx]
    
    # dp[mask] will be the set of possible XOR values we can get by partitioning the subset of bags given by mask.
    # A partition corresponds to grouping the bags (merging them) so that the final values are the sums of each group,
    # and the final answer is the XOR of these sums.
    dp = {0: {0}}  # For the empty set, the XOR is 0.
    
    # f(mask) returns the set of possible XOR outcomes for partitioning the bags indicated by mask.
    # We enforce that when partitioning a nonempty mask, we always choose a block (subset) that contains 
    # the smallest numbered bag in the current mask. This avoids counting the same partition multiple times.
    def f(mask):
        if mask in dp:
            return dp[mask]
        res = set()
        # Find the lowest bag in the current mask.
        lowbit = mask & -mask
        # Iterate over all submasks of mask that include this lowest bag.
        # We do this by iterating over all submasks (nonzero) of mask, and checking if the submask
        # actually includes the lowest bag.
        sub = mask
        while sub:
            if sub & lowbit:  # submask must include the lowest bag in 'mask'
                rem = mask - sub  # the rest of the bags after choosing one block
                # For every partition of the remainder, add the XOR of (sum of chosen block) and that outcome.
                for x in f(rem):
                    res.add(subset_sum[sub] ^ x)
            sub = (sub - 1) & mask
        dp[mask] = res
        return res

    # Compute the outcomes for the full set of bags.
    answer = len(f(total_masks - 1))
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()