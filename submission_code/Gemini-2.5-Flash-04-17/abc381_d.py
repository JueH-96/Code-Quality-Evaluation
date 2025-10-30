import sys

# Read N
# Use sys.stdin.readline for faster input
input = sys.stdin.readline
N = int(input())
# Read A
# Use list comprehension and map for faster split and int conversion
A = list(map(int, input().split()))

# max_length is initialized to 0, representing the empty sequence
max_length = 0

# Process sequences of pairs starting at even indices
# A 1122 sequence is formed by pairs (x, x), (y, y), ...
# If the first pair (x, x) starts at an even index i,
# the subsequent pairs must start at i+2, i+4, ...
# The 1122 sequence spans indices from `left` to `right + 1`, where `left` and `right`
# are the starting indices of the first and last pairs in the sequence, respectively.
# `left` and `right` must have the same parity.

# Handle even starting indices of pairs
left = 0 # `left` is the starting index of the first pair in the current valid window
counts = {} # Dictionary to store frequency of values A[k] for k in [left, right] where k is even

# Iterate through potential starting indices of the *last* pair: 0, 2, 4, ... up to N-2
# `right` represents the starting index of the current pair being considered to extend the window
for right in range(0, N - 1, 2):
    # Check if A[right] and A[right+1] form a pair
    if right + 1 < N and A[right] == A[right + 1]:
        val = A[right] # The value of the current pair
        
        # The current pair starts at index `right`. Its value is `val`.
        # For this pair to extend the current valid 1122 sequence ending at `right - 2 + 1 = right - 1`,
        # the value `val` must be distinct from the values of previous pairs
        # in the window of pair start indices [left, left+2, ..., right-2].
        # If `val` is already in `counts` with count >= 1, it means A[right] is a duplicate
        # of A[k] for some k in [left, right-2] (where k is even).
        while left <= right and val in counts and counts[val] >= 1:
            # Value A[right] is a duplicate. The distinctness condition is violated for the window [left, right].
            # We must shrink the window from the left by advancing `left` by 2,
            # until the duplicate value A[left] is removed from the window or its count becomes zero.
            remove_val = A[left]
            counts[remove_val] -= 1
            if counts[remove_val] == 0:
                del counts[remove_val]
            left += 2 # Move left pointer to the starting index of the next potential first pair
            
        # After shrinking (if needed), the window of pair start indices [left, right] satisfies the distinctness property.
        # Add the value A[right] to the frequency count for the current window [left, right].
        counts[val] = counts.get(val, 0) + 1
        
        # The current window of pair start indices [left, left+2, ..., right] corresponds to
        # a contiguous subarray of A from index `left` to `right + 1`.
        # This subarray is a valid 1122 sequence.
        # Its length is (right + 1) - left + 1 = right - left + 2.
        max_length = max(max_length, right - left + 2)
    else:
        # The current pair starting at `right` is invalid (A[right] != A[right+1] or `right + 1 >= N`).
        # This invalid pair breaks any continuous sequence of pairs starting at an even index and ending here.
        # The next potential valid sequence of pairs starting at an even index
        # must begin after `right`. The earliest possible starting index for the next pair is `right + 2`.
        left = right + 2 # Reset the `left` pointer to start searching for a new sequence after the invalid pair
        counts = {} # Clear counts for the new window


# Process sequences of pairs starting at odd indices
# Similar logic as for even indices, but starting the scan from index 1.
# The starting indices of pairs are 1, 3, 5, ...
left = 1 # `left` is the starting index of the first pair in the current valid window
counts = {} # Dictionary to store frequency of values A[k] for k in [left, right] where k is odd

# Iterate through potential starting indices of the *last* pair: 1, 3, 5, ... up to N-2
# `right` represents the starting index of the current pair being considered
for right in range(1, N - 1, 2):
    # Check if A[right] and A[right+1] form a pair
    if right + 1 < N and A[right] == A[right + 1]:
        val = A[right] # The value of the current pair
        
        # Check distinctness within the current window of pair start indices [left, left+2, ..., right].
        # If `val` is already in `counts` with count >= 1, it means A[right] is a duplicate
        # of A[k] for some k in [left, right-2] (where k is odd).
        while left <= right and val in counts and counts[val] >= 1:
            # Value A[right] is a duplicate. Shrink the window from the left.
            remove_val = A[left]
            counts[remove_val] -= 1
            if counts[remove_val] == 0:
                del counts[remove_val]
            left += 2 # Move left pointer to the starting index of the next potential first pair
            
        # After shrinking (if needed), the window of pair start indices [left, right] satisfies the distinctness property.
        # Add the value A[right] to the frequency count for the current window [left, right].
        counts[val] = counts.get(val, 0) + 1
        
        # The current window of pair start indices [left, left+2, ..., right] corresponds to
        # a contiguous subarray of A from index `left` to `right + 1`.
        # This subarray is a valid 1122 sequence.
        # Its length is (right + 1) - left + 1 = right - left + 2.
        max_length = max(max_length, right - left + 2)
    else:
        # The current pair starting at `right` is invalid (A[right] != A[right+1] or `right + 1 >= N`).
        # This invalid pair breaks any continuous sequence of pairs starting at an odd index and ending here.
        # The next potential valid sequence of pairs starting at an odd index
        # must begin after `right`. The earliest possible starting index for the next pair is `right + 2`.
        left = right + 2 # Reset the `left` pointer to start searching for a new sequence after the invalid pair
        counts = {} # Clear counts for the new window

print(max_length)