# YOUR CODE HERE
# Read the problem parameters N, K, and X from the first line of input.
# The `map(int, ...)` function applies the `int` conversion to each
# string item produced by `input().split()`.
n, k, x = map(int, input().split())

# Read the integer sequence A from the second line of input.
# This creates a list of integers.
a = list(map(int, input().split()))

# The task is to insert the integer X after the K-th element of sequence A.
# Python lists are 0-indexed, so the K-th element is at index `k-1`.
# To insert an element immediately *after* the K-th element, we need to
# insert it at index `k`.
# The `list.insert(index, element)` method does exactly this, modifying the list in-place.
a.insert(k, x)

# Print the resulting sequence.
# The `*` operator (splat/unpacking operator) passes each element of the list `a`
# as a separate argument to the `print` function.
# By default, `print` separates its arguments with a single space.
print(*a)