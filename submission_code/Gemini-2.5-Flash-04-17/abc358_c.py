import sys

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# Read S_i strings and convert them into bitmasks
# Each bit in the mask corresponds to a flavor.
# The j-th bit (1 << j) represents the availability of flavor j (0-indexed)
stand_masks = []
for _ in range(N):
    s = sys.stdin.readline().strip()
    mask = 0
    # Map flavors 0 to M-1 to bits 0 to M-1
    for j in range(M):
        if s[j] == 'o':
            mask |= (1 << j) # Set the j-th bit if flavor j (corresponding to S_i[j]) is sold
    stand_masks.append(mask)

# The target mask is one where all bits from 0 to M-1 are set,
# indicating that all M flavors are covered.
all_flavors_mask = (1 << M) - 1

# Initialize minimum number of stands needed.
# Start with a value larger than any possible valid answer (the maximum possible is N).
min_stands = N + 1 

# Iterate through all possible non-empty subsets of stands.
# A subset of the N stands can be represented by an N-bit binary number (a bitmask).
# If the i-th bit of the subset_mask is set, it means stand i (0-indexed) is included in the subset.
# We iterate from 1 up to 2^N - 1 to cover all non-empty subsets.
for subset_mask in range(1, (1 << N)):
    covered_flavors_mask = 0
    num_stands_in_subset = 0

    # Check which stands are included in the current subset
    # Iterate through each stand index i from 0 to N-1
    for i in range(N):
        # If the i-th bit is set in subset_mask, stand i (0-indexed) is included in the current subset
        if (subset_mask >> i) & 1:
            # Combine the flavors sold by stand i with the currently covered flavors using bitwise OR
            covered_flavors_mask |= stand_masks[i]
            # Increment the count of stands in the current subset
            num_stands_in_subset += 1

    # Check if this subset covers all flavors
    if covered_flavors_mask == all_flavors_mask:
        # If it covers all flavors, this is a valid subset of stands.
        # Update the minimum number of stands found so far with the size of this subset.
        min_stands = min(min_stands, num_stands_in_subset)

# Print the result, which is the minimum number of stands found to cover all flavors.
print(min_stands)