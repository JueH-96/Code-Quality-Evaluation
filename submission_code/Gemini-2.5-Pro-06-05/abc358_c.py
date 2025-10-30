import sys
import itertools

# Read the number of stands (N) and flavors (M).
N, M = map(int, sys.stdin.readline().split())

# For each stand, create a bitmask to represent the flavors it sells.
# The j-th bit (0-indexed) is 1 if flavor j+1 is available, 0 otherwise.
stand_masks = []
for _ in range(N):
    s = sys.stdin.readline().strip()
    mask = 0
    for j in range(M):
        if s[j] == 'o':
            # Set the j-th bit to 1.
            mask |= (1 << j)
    stand_masks.append(mask)

# The target mask represents having all M flavors. This is an integer
# with the first M bits set to 1 (e.g., for M=5, target is 31 or binary 11111).
target_mask = (1 << M) - 1

# Iterate through the number of stands to visit, k, from 1 to N.
# We check in increasing order of k to find the minimum.
for k in range(1, N + 1):
    # Generate all combinations of k stands. Instead of using stand indices,
    # we can directly generate combinations of their pre-calculated masks.
    for combo_of_masks in itertools.combinations(stand_masks, k):
        
        # Calculate the union of flavors for this combination of stands
        # by performing a bitwise OR on their masks.
        combined_mask = 0
        for mask in combo_of_masks:
            combined_mask |= mask
            
        # If the combined mask matches the target, we have found a solution
        # with k stands. Since we are checking k in increasing order, this
        # is the minimum number of stands required.
        if combined_mask == target_mask:
            print(k)
            # Exit the program immediately as the minimum is found.
            exit()