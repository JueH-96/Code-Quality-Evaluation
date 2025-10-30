import sys

# Read N
N = int(sys.stdin.readline())

# Read the array A
# Convert to 0-based indexing internally
A = list(map(int, sys.stdin.readline().split()))

# --- Calculate distinct count for all prefixes A[0...i] ---
# distinct_left[i] will store the number of distinct elements in A[0...i]
distinct_left = [0] * N
# Use a boolean array to track seen elements in the current prefix.
# Since 1 <= A_i <= N, we can use an array of size N+1 for direct indexing.
# seen[val] is True if 'val' has been seen in the current prefix
seen = [False] * (N + 1)
count = 0 # Current distinct count for the prefix
for i in range(N):
    val = A[i]
    if not seen[val]:
        seen[val] = True
        count += 1
    distinct_left[i] = count

# --- Calculate distinct count for all suffixes A[i...N-1] ---
# distinct_right[i] will store the number of distinct elements in A[i...N-1]
distinct_right = [0] * N
# Reset the seen array for calculating suffix distinct counts
seen = [False] * (N + 1)
count = 0 # Current distinct count for the suffix
# Iterate backward from the end of the array
for i in range(N - 1, -1, -1):
    val = A[i]
    if not seen[val]:
        seen[val] = True
        count += 1
    distinct_right[i] = count

# --- Calculate the maximum sum of distinct counts for all possible split points ---
# The problem defines split point 'i' (1-based, 1 <= i <= N-1)
# The split is between A[i-1] and A[i] (0-based indexing)
# Left subarray: A[0...i-1]. Distinct count is distinct_left[i-1].
# Right subarray: A[i...N-1]. Distinct count is distinct_right[i].
max_sum = 0
# Iterate through all valid split points 'i' from 1 to N-1
# 'i' corresponds to the index *after* the split in 0-based indexing
# e.g., i=1 means split after index 0, left = A[0], right = A[1...N-1]
# i=N-1 means split after index N-2, left = A[0...N-2], right = A[N-1]
for i in range(1, N):
    left_distinct = distinct_left[i - 1] # Distinct count for A[0]...A[i-1]
    right_distinct = distinct_right[i]   # Distinct count for A[i]...A[N-1]
    current_sum = left_distinct + right_distinct
    max_sum = max(max_sum, current_sum)

# Print the result
print(max_sum)