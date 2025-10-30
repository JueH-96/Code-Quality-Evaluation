# YOUR CODE HERE

import sys

# Read the number of queries
Q = int(sys.stdin.readline().strip())

# Initialize an empty list to store the sequence
A = []

# Process the queries
for _ in range(Q):
    query = sys.stdin.readline().strip().split()
    if query[0] == '1':
        A.append(int(query[1]))
    else:
        k = int(query[1])
        print(A[-k])