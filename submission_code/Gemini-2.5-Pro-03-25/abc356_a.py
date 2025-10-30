# YOUR CODE HERE
import sys

# Read N, L, R from standard input
# N: the length of the sequence
# L: the starting position of the segment to reverse (1-based index)
# R: the ending position of the segment to reverse (1-based index)
N, L, R = map(int, sys.stdin.readline().split())

# Create the initial sequence A = (1, 2, ..., N)
# `range(1, N + 1)` generates numbers from 1 up to N (inclusive).
# `list()` converts the range object into a list.
A = list(range(1, N + 1))

# Adjust L and R to 0-based indices for Python slicing
# In Python, list indices start at 0.
# The L-th element is at index L-1.
# The R-th element is at index R-1.
# A slice `A[start:end]` includes elements from index `start` up to `end-1`.
# To reverse the elements from the L-th position to the R-th position (inclusive),
# we need the slice starting at index `L-1` and ending just after index `R-1`.
# Therefore, the starting index for the slice is `L-1`.
# The ending index for the slice (exclusive) is `R`.
start_index = L - 1
end_index = R 

# Reverse the specified segment in place using slicing assignment.
# `A[start_index:end_index]` selects the segment of the list from the L-th element to the R-th element.
# `[::-1]` is a slicing technique that creates a reversed copy of a sequence.
# The assignment operation `A[start_index:end_index] = ...` replaces the original segment 
# in the list `A` with the reversed version.
# If L=R, the slice contains one element, and reversing it has no effect, which is correct.
A[start_index:end_index] = A[start_index:end_index][::-1]

# Print the final sequence with elements separated by spaces.
# The `*` operator unpacks the list `A` into individual arguments for the `print` function.
# By default, `print()` separates its arguments with a space and adds a newline character at the end.
print(*A)