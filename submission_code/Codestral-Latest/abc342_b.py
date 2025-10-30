# Read the number of people
N = int(input())

# Read the positions of people
positions = list(map(int, input().split()))

# Read the number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    A, B = map(int, input().split())
    # Find the positions of A and B
    pos_A = positions.index(A)
    pos_B = positions.index(B)
    # Determine who is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)