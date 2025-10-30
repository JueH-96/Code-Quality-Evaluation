# Read the number of people
N = int(input())
# Read the list of people's positions
P = list(map(int, input().split()))
# Read the number of queries
Q = int(input())
# Process each query
for _ in range(Q):
    A, B = map(int, input().split())
    # Find the positions of A and B in the list
    pos_A = P.index(A)
    pos_B = P.index(B)
    # Determine which person is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)