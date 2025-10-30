import sys

# Read N
N = int(sys.stdin.readline())

# Read array A
A = list(map(int, sys.stdin.readline().split()))

# Calculate distinct counts for all possible left subarrays A[0...i-1]
# left_distincts[i] will store distinct count of A[0...i-1] for i=1...N-1
# We use a list of size N and fill indices from 1 to N-1
# left_distincts[i] corresponds to the distinct count of the prefix of length i
left_distincts = [0] * N
current_set = set()
# Iterate through the array from left to right, building prefixes
# k is the 0-based index of the current element being added
# The prefix A[0...k] has distinct count len(current_set) after adding A[k]
# This prefix A[0...k] is the left part for the split point i = k+1 (1-based)
for k in range(N - 1): # k goes from 0 to N-2
    current_set.add(A[k])
    # Store the distinct count of A[0...k] at index k+1, corresponding to split point i=k+1
    left_distincts[k + 1] = len(current_set)

# Calculate distinct counts for all possible right subarrays A[i...N-1]
# right_distincts[i] will store distinct count of A[i...N-1] for i=1...N-1
# We use a list of size N and fill indices from 1 to N-1
# right_distincts[i] corresponds to the distinct count of the suffix starting at index i (0-based)
right_distincts = [0] * N
current_set = set()
# Iterate through the array from right to left, building suffixes
# k is the 0-based index of the current element being added
# The suffix A[k...N-1] has distinct count len(current_set) after adding A[k]
# This suffix A[k...N-1] is the right part for the split point i = k (1-based)
for k in range(N - 1, 0, -1): # k goes from N-1 down to 1
    current_set.add(A[k])
    # Store the distinct count of A[k...N-1] at index k, corresponding to split point i=k
    right_distincts[k] = len(current_set)

# Find the maximum possible sum of distinct counts
max_sum = 0
# The split point i (1-based) goes from 1 to N-1
# For a split at i (1-based), the left part is A[0...i-1] and the right part is A[i...N-1] (0-based)
for i in range(1, N):
    # Distinct count of left part A[0...i-1] is left_distincts[i]
    # Distinct count of right part A[i...N-1] is right_distincts[i]
    current_sum = left_distincts[i] + right_distincts[i]
    max_sum = max(max_sum, current_sum)

# Print the result
print(max_sum)