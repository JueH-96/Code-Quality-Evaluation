# YOUR CODE HERE
import sys

# Read N from standard input
# N represents the number of colors, and 2N is the total number of people.
n = int(sys.stdin.readline())

# Read the list A of 2*N colors from standard input.
# A[i] is the color of the person at the (i+1)-th position from the left (using 0-based indexing for the list).
a = list(map(int, sys.stdin.readline().split()))

# Initialize a counter to store the number of colors that satisfy the condition.
count = 0

# The condition is that there is exactly one person between the two people wearing clothes of color i.
# This means if the two people wearing color i are at 1-based positions p1 and p2 (p1 < p2),
# then the number of people between them is p2 - p1 - 1.
# We want p2 - p1 - 1 = 1, which simplifies to p2 - p1 = 2.
#
# In terms of 0-based indices j1 and j2 (j1 < j2), where p1 = j1 + 1 and p2 = j2 + 1,
# the condition becomes (j2 + 1) - (j1 + 1) = 2, which means j2 - j1 = 2.
# This means we are looking for pairs of indices (j, j+2) such that A[j] == A[j+2].
#
# We can iterate through the list A and check this condition directly.
# The list `a` has length 2*N. Indices range from 0 to 2*N - 1.
# We need to compare `a[i]` and `a[i+2]`.
# The loop index `i` should go from 0 up to the maximum value such that `i+2` is a valid index.
# The maximum index in `a` is `2*n - 1`.
# So, the maximum value for `i+2` must be `2*n - 1`.
# This implies the maximum value for `i` is `(2*n - 1) - 2 = 2*n - 3`.
# The `range(stop)` function generates integers from 0 up to `stop - 1`.
# To make the maximum value of `i` be `2*n - 3`, the `stop` value must be `(2*n - 3) + 1 = 2*n - 2`.
for i in range(2 * n - 2):
    # Check if the element at index `i` is equal to the element at index `i+2`.
    if a[i] == a[i+2]:
        # If they are equal, it means a pair of people wearing the same color (color a[i])
        # has exactly one person (at index i+1) between them.
        # We increment the count.
        # The problem statement guarantees that each color from 1 to N appears exactly twice.
        # Therefore, if a color satisfies the condition (its occurrences are at indices j and j+2),
        # this loop will find that specific pair exactly once when i = j.
        # Thus, incrementing the count here correctly counts the number of *colors* satisfying the condition.
        count += 1

# Print the final count to standard output
print(count)