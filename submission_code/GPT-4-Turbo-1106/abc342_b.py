# Read the number of people and their positions
N = int(input())
positions = list(map(int, input().split()))

# Read the number of queries
Q = int(input())
queries = [tuple(map(int, input().split())) for _ in range(Q)]

# Process each query
for A, B in queries:
    # Find the index of A and B in the positions list
    index_A = positions.index(A)
    index_B = positions.index(B)
    # Print the person number of the person standing further to the front
    if index_A < index_B:
        print(A)
    else:
        print(B)