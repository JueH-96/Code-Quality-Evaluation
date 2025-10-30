# Read N, L, R from standard input
N, L, R = map(int, input().split())

# Create the initial sequence A = (1, 2, ..., N)
# In Python, lists are 0-indexed, so range(1, N + 1) generates numbers from 1 to N.
A = list(range(1, N + 1))

# Adjust L and R for 0-based indexing for slicing
# The L-th element corresponds to index L-1
# The R-th element corresponds to index R-1
# A slice A[start:end] includes elements from index 'start' up to 'end-1'.
# So, to include the R-th element (at index R-1), the 'end' of the slice should be R.
l_idx = L - 1
r_idx = R

# Reverse the sub-sequence from index l_idx to r_idx-1 (inclusive)
# Python allows direct assignment to a slice using a reversed slice of itself.
# A[l_idx:r_idx] extracts the sub-list.
# [::-1] reverses that extracted sub-list.
# The result is then assigned back to the original slice in A.
A[l_idx:r_idx] = A[l_idx:r_idx][::-1]

# Print the final sequence.
# The *A unpacks the list elements as separate arguments to print,
# which by default prints them separated by spaces.
print(*A)