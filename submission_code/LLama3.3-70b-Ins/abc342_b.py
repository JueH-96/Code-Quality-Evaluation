import sys

# Read the number of people
N = int(input())

# Read the order of people
P = list(map(int, input().split()))

# Read the number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    # Read the query
    A, B = map(int, input().split())

    # Find the positions of A and B in the order
    pos_A = P.index(A) + 1
    pos_B = P.index(B) + 1

    # Print the person number of the person standing further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)