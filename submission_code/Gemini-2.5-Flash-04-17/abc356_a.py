import sys

# Read the three integers N, L, and R from standard input.
# They are provided on a single line, space-separated.
# N: The total number of elements in the sequence.
# L: The 1-based index of the starting element for the reversal (inclusive).
# R: The 1-based index of the ending element for the reversal (inclusive).
N, L, R = map(int, sys.stdin.readline().split())

# Create the initial sequence A = (1, 2, ..., N).
# In Python, lists are 0-indexed. The numbers 1 through N are stored in a list.
# For example, if N=5, the list will be [1, 2, 3, 4, 5].
sequence = list(range(1, N + 1))

# Perform the reversal operation.
# The problem asks to reverse elements from the L-th to the R-th position.
# Since Python uses 0-based indexing:
# The L-th element is located at index L-1.
# The R-th element is located at index R-1.
# The slice in Python that includes elements from index L-1 up to R-1 is sequence[L-1 : (R-1)+1],
# which simplifies to sequence[L-1 : R].
# We reverse this specific slice using the slicing step [::-1].
# The result of the reversed slice is then assigned back to the original slice
# in the sequence list, effectively performing the reversal in-place.
sequence[L-1:R] = sequence[L-1:R][::-1]

# Print the final sequence after the reversal operation.
# The '*' operator is used to unpack the elements of the list 'sequence'
# as separate arguments to the print function.
# By default, print() separates arguments with spaces and ends with a newline,
# matching the required output format.
print(*sequence)