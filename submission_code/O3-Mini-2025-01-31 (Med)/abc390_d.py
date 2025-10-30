def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    N = 1 << n  # total number of masks
    # Precompute sum for each subset (mask) of the n indices.
    sums = [0] * N
    for mask in range(1, N):
        lb = mask & -mask  # lowest set bit
        i = lb.bit_length() - 1  # index of lowest set bit
        prev = mask - lb
        sums[mask] = sums[prev] + A[i]
    
    # dp[mask] will store the set of possible XOR values from partitioning the subset represented by mask.
    # A partition corresponds to grouping some indices together so that each group contributes its sum,
    # and the final XOR is the XOR over the sums of groups.
    dp = [set() for _ in range(N)]
    dp[0].add(0)  # empty partition: XOR is 0.
    
    # To avoid repeated counting in partitions (since the order of groups does not matter)
    # we fix that in every partition we require that the block containing the smallest-index element in the mask
    # is chosen in a canonical way.
    #
    # For each nonempty mask, let i be the smallest index present.
    # Then, for every nonempty subset X of mask that contains i, we combine the block X (with sum sums[X])
    # and a partition of the remaining indices (mask - X). The overall XOR is: sums[X] XOR v,
    # where v is a possible XOR from partitioning mask - X.
    #
    # We iterate in increasing order of the number of bits in mask.
    masks = list(range(N))
    masks.sort(key=lambda x: x.bit_count())
    
    for mask in masks:
        if mask == 0:
            continue
        # Get the smallest index in mask.
        lowbit = mask & -mask  # lowest set bit
        i = lowbit.bit_length() - 1
        
        # Enumerate all nonempty submasks X of mask that include index i.
        # Standard trick: iterate over all submasks of mask.
        sub = mask
        while sub:
            if sub & (1 << i):  # ensure X includes the smallest element
                rem = mask - sub  # remainder of indices after taking group X
                base = sums[sub]
                # For each possible XOR value from partitioning the remainder, add the XOR with group sum.
                for v in dp[rem]:
                    dp[mask].add(base ^ v)
            sub = (sub - 1) & mask
    
    # The answer is the number of distinct XOR values for the entire set (mask with all indices).
    ans = len(dp[N - 1])
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()