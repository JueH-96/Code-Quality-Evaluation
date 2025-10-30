# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    stand_flavors_masks = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        
        # Convert string S_i to a bitmask.
        # The j-th character of S_i corresponds to the j-th flavor (0-indexed).
        # If S_i[j] is 'o', set the j-th bit in the mask.
        current_mask = 0
        for j in range(M):
            if s[j] == 'o':
                current_mask |= (1 << j)
        stand_flavors_masks.append(current_mask)

    # The target mask represents all M flavors being covered.
    # For M flavors (0 to M-1), this is a binary number with all M bits set.
    # E.g., if M=5, target is 11111_2 = 31.
    all_flavors_mask = (1 << M) - 1

    # Initialize min_stands with a value that is impossible to beat,
    # e.g., N+1, since the maximum number of stands is N.
    min_stands = N + 1 

    # Iterate through all possible subsets of stands.
    # There are 2^N possible subsets. Each integer 'i' from 0 to 2^N - 1
    # represents a unique subset.
    # If the k-th bit of 'i' is set, it means stand 'k' is included in the current subset.
    for i in range(1 << N): # i iterates from 0 to 2^N - 1
        current_combined_flavors = 0 # This mask will accumulate flavors from selected stands
        stands_count = 0             # Count of stands in the current subset
        
        # Check each stand (from 0 to N-1) to see if it's part of the current subset 'i'.
        for k in range(N):
            # Check if the k-th bit of 'i' is set
            if (i >> k) & 1: 
                # If stand 'k' is selected, add its flavors to the combined mask
                current_combined_flavors |= stand_flavors_masks[k]
                stands_count += 1
        
        # After evaluating all stands for the current subset 'i',
        # check if this subset covers all flavors.
        if current_combined_flavors == all_flavors_mask:
            # If it does, update min_stands if this subset is smaller
            min_stands = min(min_stands, stands_count)
            
    # Print the minimum number of stands required.
    sys.stdout.write(str(min_stands) + "
")

solve()