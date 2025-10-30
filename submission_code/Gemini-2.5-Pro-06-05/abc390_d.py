import sys

def main():
    """
    Solves the problem using dynamic programming on subsets.
    """
    try:
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handles potential empty input on some platforms.
        return

    # Precompute sums of A_i for all 2^N subsets (masks) for efficiency.
    sums = [0] * (1 << N)
    for i in range(N):
        for mask in range(1 << i):
            sums[mask | (1 << i)] = sums[mask] + A[i]

    # dp[mask] will store the set of possible XOR sums for partitions of the bags in `mask`.
    dp = [set() for _ in range(1 << N)]
    
    # Base case: for an empty set of bags (mask=0), the only partition is the empty one.
    # The XOR sum of an empty set of numbers is conventionally 0.
    dp[0] = {0}

    # Iterate through all non-empty subsets of bags, represented by masks.
    for mask in range(1, 1 << N):
        # To build partitions of `mask` without overcounting, we fix one element
        # and iterate through all groups containing it. A standard choice is the
        # element corresponding to the least significant bit (LSB) of the mask.
        
        lsb_idx = (mask & -mask).bit_length() - 1
        
        # We will form a group `group1` that includes the LSB element.
        # The remaining part of `mask` for forming this group is `rest_of_mask`.
        rest_of_mask = mask ^ (1 << lsb_idx)
        
        # Iterate through all subsets `sub` of `rest_of_mask`.
        # Each `sub` combined with the LSB element forms a potential first group.
        sub = rest_of_mask
        while True:
            # The first group in our partition of `mask`.
            group1_mask = sub | (1 << lsb_idx)
            
            # The rest of the elements in `mask` form the second part of the partition.
            group2_mask = mask ^ group1_mask
            
            # The sum of stones in the first group.
            group1_sum = sums[group1_mask]
            
            # New possible XOR sums are formed by taking `group1_sum` and XORing it
            # with every possible XOR sum from partitions of the remaining elements.
            for xor_val in dp[group2_mask]:
                dp[mask].add(group1_sum ^ xor_val)

            # This idiom iterates through all submasks of `rest_of_mask`.
            if sub == 0:
                break
            sub = (sub - 1) & rest_of_mask
            
    # The final answer is the number of unique XOR sums for the set of all bags.
    print(len(dp[(1 << N) - 1]))

if __name__ == "__main__":
    main()