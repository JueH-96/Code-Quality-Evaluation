import sys

def lowest_set_bit_index(mask):
    """
    Finds the 0-indexed position of the lowest set bit in a mask.
    e.g., mask = 12 (binary 1100), returns 2.
    mask = 7 (binary 0111), returns 0.
    Returns -1 for mask = 0.
    """
    if mask == 0:
        return -1
    idx = 0
    # Check bits from right to left until a set bit is found
    while not (mask >> idx) & 1:
        idx += 1
    return idx

def solve():
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute sum of A_i for each subset mask
    # S_A[mask] will store the sum of A[i] for all indices i
    # where the i-th bit of mask is set.
    # Mask 0 represents the empty set, sum is 0.
    S_A = [0] * (1 << N)
    for mask in range(1 << N):
        if mask == 0:
            continue
        # The current mask includes the lowest set bit, say at index i.
        # The sum for mask is A[i] plus the sum for the mask excluding bit i.
        i = lowest_set_bit_index(mask)
        S_A[mask] = A[i] + S_A[mask ^ (1 << i)]

    # DP state: dp[mask] is the set of possible bitwise XOR sums
    # of sums of partitions, considering only the elements corresponding
    # to the set bits in 'mask'.
    dp = [set() for _ in range(1 << N)]
    dp[0] = {0} # Base case: The empty set of elements has an XOR sum of 0.

    # Iterate through masks from 1 up to 2^N - 1.
    # Processing masks in increasing order ensures that when computing dp[mask],
    # the sets dp[complement_mask] for any complement_mask with fewer set bits
    # have already been computed.
    for mask in range(1 << N):
        if mask == 0:
            continue # Base case already handled

        # Find the index of the lowest set bit in the current mask.
        # This element 'i' must belong to exactly one set in any valid partition
        # of the elements represented by 'mask'.
        i = lowest_set_bit_index(mask)
        first_bit_mask = 1 << i

        # We iterate through all possible sets 'submask' that contain element 'i'
        # and are a subset of 'mask'. 'submask' represents one part of the partition
        # that contains element 'i'.
        # A submask containing bit 'i' is formed by (1 << i) OR any submask of (mask ^ (1 << i)).
        
        # 'rest_mask_template' represents the elements in 'mask' excluding element 'i'.
        rest_mask_template = mask ^ first_bit_mask
        
        # Iterate through all submasks 's_rest' of 'rest_mask_template'.
        # Each 's_rest' combined with 'first_bit_mask' gives a possible 'submask'
        # that contains 'i'.
        s_rest = rest_mask_template
        while True:
            # Construct the 'submask' containing element 'i' and elements from 's_rest'.
            submask = first_bit_mask | s_rest
            
            # 'complement_mask' represents the elements in 'mask' that are NOT in 'submask'.
            # These elements must be partitioned independently.
            # complement_mask = mask ^ submask is equivalent to rest_mask_template ^ s_rest
            complement_mask = mask ^ submask

            # The sum of stones for the group represented by 'submask' is S_A[submask].
            # We combine this sum with every possible XOR sum obtained from partitioning
            # the remaining elements ('complement_mask').
            sum_of_submask = S_A[submask]
            for x in dp[complement_mask]:
                dp[mask].add(sum_of_submask ^ x)

            # Move to the next submask of 'rest_mask_template'.
            # This standard trick iterates through submasks in decreasing order.
            if s_rest == 0:
                break # Finished iterating all submasks
            s_rest = (s_rest - 1) & rest_mask_template

    # The set dp[(1 << N) - 1] contains all possible XOR sums when partitioning all N bags.
    # The size of this set is the number of different possible XOR values.
    print(len(dp[(1 << N) - 1]))

solve()