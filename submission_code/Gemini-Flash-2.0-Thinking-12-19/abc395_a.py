# YOUR CODE HERE
import sys

# Read the number of elements
# N is guaranteed to be an integer between 2 and 100.
n = int(sys.stdin.readline())

# Read the sequence of integers
# The sequence A is given on a single line, space-separated.
# We split the line into strings, convert each string to an integer,
# and store them in a list called 'a'.
a = list(map(int, sys.stdin.readline().split()))

# Determine if the sequence is strictly increasing.
# A sequence A = (A_1, A_2, ..., A_N) is strictly increasing if
# A_i < A_{i+1} for every integer i with 1 <= i < N.
# In 0-based indexing used in Python lists, this means checking
# a[i] < a[i+1] for all indices i from 0 up to N-2.
# The range(n - 1) generates the required indices: 0, 1, ..., n-2.
# We can use a generator expression (a[i] < a[i+1] for i in range(n - 1))
# to check the condition for each pair of adjacent elements.
# This expression produces a sequence of boolean values.
# The built-in all() function returns True if all elements in the iterable
# (the sequence of boolean values from the generator) are True.
# If any comparison a[i] < a[i+1] is False (meaning a[i] >= a[i+1]),
# the all() function will immediately return False, indicating the sequence
# is not strictly increasing.
is_strictly_increasing = all(a[i] < a[i+1] for i in range(n - 1))

# Print the result.
# If is_strictly_increasing is True, print "Yes".
# Otherwise (if it's False), print "No".
# The judge accepts "Yes" or "No" regardless of case.
if is_strictly_increasing:
    print("Yes")
else:
    print("No")