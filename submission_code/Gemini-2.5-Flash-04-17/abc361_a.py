# YOUR CODE HERE
# Read N, K, X from the first line
n, k, x = map(int, input().split())

# Read the sequence A from the second line
a = list(map(int, input().split()))

# Insert X immediately after the K-th element.
# The problem uses 1-based indexing for the K-th element.
# In Python (0-based indexing), the K-th element is at index K-1.
# We need to insert X *after* the element at index K-1.
# This means inserting X at index K.
# list.insert(index, value) inserts the value before the element currently at the given index.
# So, a.insert(k, x) will insert x at index k.
# If the original list indices were 0, 1, ..., K-1, K, ..., N-1,
# after inserting at index K, the new list indices will be 0, 1, ..., K-1, K, K+1, ..., N.
# The element originally at index K-1 is still at index K-1.
# The new element X is at index K.
# The element originally at index K is now at index K+1, and so on.
# This correctly places X immediately after the K-th element (at index K-1).
a.insert(k, x)

# Print the resulting sequence B.
# The * operator unpacks the list a into arguments for the print function.
# By default, print separates arguments with spaces.
print(*a)