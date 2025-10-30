# Read the input
N, Q = map(int, input().split())
x = list(map(int, input().split()))

# Initialize the sequence A and the set S
A = [0] * N
S = set()

# Process the queries
for i in range(Q):
    if x[i] in S:
        S.remove(x[i])
    else:
        S.add(x[i])
    
    for j in range(N):
        if j+1 in S:
            A[j] += len(S)

# Print the final sequence A
print(" ".join(map(str, A)))