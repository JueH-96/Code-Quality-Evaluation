# Read the input
N = int(input())
P = [int(x) for x in input().split()]
Q = int(input())
queries = []
for _ in range(Q):
    A, B = [int(x) for x in input().split()]
    queries.append((A, B))

# Solve the problem
for A, B in queries:
    # Find the positions of A and B
    a_pos = P.index(A)
    b_pos = P.index(B)
    
    # Print the person number of the person standing further to the front
    if a_pos < b_pos:
        print(P[a_pos])
    else:
        print(P[b_pos])