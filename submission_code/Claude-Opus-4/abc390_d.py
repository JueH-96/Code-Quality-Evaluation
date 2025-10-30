def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # For each subset (represented as bitmask), calculate sum of stones
    subset_sums = {}
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += a[i]
        subset_sums[mask] = total
    
    # Set to store all possible XOR values
    possible_xors = set()
    
    # Try all possible partitions
    # We'll use dynamic programming approach
    # dp[mask] = set of possible XOR values for the subset represented by mask
    dp = [set() for _ in range(1 << n)]
    dp[0].add(0)
    
    for mask in range(1 << n):
        if not dp[mask]:
            continue
            
        # Try adding each possible subset that doesn't overlap with current mask
        submask = ((1 << n) - 1) ^ mask
        subset = submask
        while subset > 0:
            # For each subset, we can add its sum to our current XOR values
            for xor_val in dp[mask]:
                new_mask = mask | subset
                new_xor = xor_val ^ subset_sums[subset]
                dp[new_mask].add(new_xor)
            subset = (subset - 1) & submask
    
    # The answer is all possible XOR values when all bags are considered
    print(len(dp[(1 << n) - 1]))

solve()