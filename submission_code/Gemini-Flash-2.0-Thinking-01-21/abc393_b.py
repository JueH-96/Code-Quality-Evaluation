# YOUR CODE HERE
import sys

# Read the input string from stdin
S = input()

count = 0
n = len(S)

# Iterate through all possible indices for the first character 'A'
# The first character 'A' must be at index idx_i (0-based), where 0 <= idx_i.
# To form a triple (idx_i, idx_j, idx_k) with idx_i < idx_j < idx_k and equal difference,
# the minimum possible indices are 0, 1, 2 (common difference d=1).
# This means idx_i must be less than n-2 (to allow for at least two indices after it).
# So, idx_i can range from 0 up to n - 3.
for idx_i in range(n - 2):
    # If the character at idx_i is not 'A', it cannot be the start of a valid triple.
    # We can skip checking this idx_i entirely.
    if S[idx_i] != 'A':
        continue

    # Iterate through all possible indices for the third character 'C'
    # The third character 'C' must be at index idx_k (0-based), where idx_k <= n - 1.
    # The equal difference condition idx_j - idx_i = idx_k - idx_j implies idx_j = (idx_i + idx_k) / 2.
    # For idx_j to be strictly between idx_i and idx_k, we need idx_k >= idx_i + 2.
    # (If idx_k = idx_i + 1, (idx_i + idx_k) would be 2*idx_i + 1, which is odd, no integer idx_j.
    # If idx_k = idx_i, this violates idx_i < idx_k)
    # So, idx_k must be at least idx_i + 2 and at most n - 1.
    for idx_k in range(idx_i + 2, n):
        # If the character at idx_k is not 'C', it cannot be the end of a valid triple.
        # We can skip checking this idx_k.
        if S[idx_k] != 'C':
            continue

        # For the indices to form an arithmetic progression (equal interval),
        # the sum of the first and third indices (idx_i + idx_k) must be an even number
        # so that the middle index idx_j = (idx_i + idx_k) / 2 is an integer.
        if (idx_i + idx_k) % 2 == 0:
            # Calculate the potential middle index idx_j.
            # Integer division is used.
            idx_j = (idx_i + idx_k) // 2

            # Check if the character at the calculated middle index idx_j is 'B'.
            # We already ensured idx_i < idx_j < idx_k by the loop ranges for idx_i and idx_k
            # (idx_k >= idx_i + 2) and the check that (idx_i + idx_k) is even.
            # For example, if idx_k = idx_i + 2 (minimum even difference), idx_j = (idx_i + idx_i + 2) / 2 = idx_i + 1.
            # This gives idx_i, idx_i+1, idx_i+2, which are in increasing order with difference 1.

            if S[idx_j] == 'B':
                # We have found a valid triple (idx_i, idx_j, idx_k) in 0-based indices.
                # This corresponds directly to a valid triple in 1-based indices.
                count += 1

# Print the final count to stdout
print(count)