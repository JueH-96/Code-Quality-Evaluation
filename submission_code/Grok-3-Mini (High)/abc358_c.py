import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N and M
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Create a list to store the flavor bitmasks for each stand
flav_masks = []
full_mask = (1 << M) - 1  # Bitmask with all flavors set

# Convert each stand's string to a bitmask
for _ in range(N):
    s = data[index]
    index += 1
    mask = 0
    for j in range(M):
        if s[j] == 'o':
            mask |= (1 << j)  # Set the j-th bit if flavor is sold
    flav_masks.append(mask)

# Find the minimum number of stands that cover all flavors
min_count = N + 1  # Initialize to a value larger than possible maximum

# Iterate over all non-empty subsets of stands using bitmasks
for stand_subset_mask in range(1, 1 << N):  # From 1 to 2^N - 1
    or_result = 0  # Bitmask for the union of flavors in this subset
    for i in range(N):
        if (stand_subset_mask & (1 << i)) != 0:  # If stand i is in the subset
            or_result |= flav_masks[i]  # OR with the flavor mask of stand i
    if or_result == full_mask:  # If all flavors are covered
        # Calculate the number of stands in this subset
        size = bin(stand_subset_mask).count('1')
        if size < min_count:
            min_count = size  # Update minimum count

# Output the minimum number of stands
print(min_count)