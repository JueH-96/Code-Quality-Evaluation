import sys

# Read N
N = int(sys.stdin.readline())
# Read A as a list of integers
A = list(map(int, sys.stdin.readline().split()))

# Compute prefix_distinct array
# prefix_distinct[k] will store the number of distinct elements in the subarray A[0...k]
prefix_distinct = [0] * N
# 'seen_prefix' is a boolean array to keep track of elements encountered in the current prefix
# The size is N+1 because array elements A_i are guaranteed to be between 1 and N
seen_prefix = [False] * (N + 1)
current_distinct_count = 0
# Iterate through the array from left to right
for k in range(N):
    element = A[k]
    # If the current element has not been seen in the prefix before
    if not seen_prefix[element]:
        # Mark the element as seen
        seen_prefix[element] = True
        # Increment the count of distinct elements
        current_distinct_count += 1
    # Store the total distinct count for the subarray A[0...k]
    prefix_distinct[k] = current_distinct_count

# Compute suffix_distinct array
# suffix_distinct[k] will store the number of distinct elements in the subarray A[k...N-1]
suffix_distinct = [0] * N
# 'seen_suffix' is a boolean array to keep track of elements encountered in the current suffix
# The size is N+1 because array elements A_i are guaranteed to be between 1 and N
seen_suffix = [False] * (N + 1)
current_distinct_count = 0
# Iterate through the array from right to left
for k in range(N - 1, -1, -1):
    element = A[k]
    # If the current element has not been seen in the suffix before
    if not seen_suffix[element]:
        # Mark the element as seen
        seen_suffix[element] = True
        # Increment the count of distinct elements
        current_distinct_count += 1
    # Store the total distinct count for the subarray A[k...N-1]
    suffix_distinct[k] = current_distinct_count

# Find the maximum possible sum of distinct counts when splitting the array
# The problem asks to split the array at position i (1-based), where 1 <= i <= N-1.
# This corresponds to splitting into A[0...i-1] and A[i...N-1] (0-based indexing).
# The number of distinct elements in A[0...i-1] is prefix_distinct[i-1].
# The number of distinct elements in A[i...N-1] is suffix_distinct[i].
max_sum = 0
# Iterate through all possible split points 'i' from 1 to N-1
# The loop range `range(1, N)` covers i = 1, 2, ..., N-1
for i in range(1, N):
    # Calculate the sum of distinct counts for the split at position 'i'
    # The first subarray is A[0...i-1], its distinct count is prefix_distinct[i-1]
    # The second subarray is A[i...N-1], its distinct count is suffix_distinct[i]
    current_sum = prefix_distinct[i - 1] + suffix_distinct[i]
    # Update the maximum sum found so far
    max_sum = max(max_sum, current_sum)

# Print the final maximum sum
print(max_sum)