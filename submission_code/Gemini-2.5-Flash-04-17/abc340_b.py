# YOUR CODE HERE
import sys

# Read the number of queries
Q = int(sys.stdin.readline())

# Initialize an empty list
A = []

# Process each query
for _ in range(Q):
    query = list(map(int, sys.stdin.readline().split()))

    query_type = query[0]

    if query_type == 1:
        # Type 1 query: append x
        x = query[1]
        A.append(x)
    elif query_type == 2:
        # Type 2 query: find k-th from end
        k = query[1]
        # The k-th element from the end is at index -k in Python.
        # k is guaranteed to be a positive integer and not greater than the current length of A.
        # For example, k=1 corresponds to index -1 (last element), k=2 to index -2 (second last), etc.
        print(A[-k])