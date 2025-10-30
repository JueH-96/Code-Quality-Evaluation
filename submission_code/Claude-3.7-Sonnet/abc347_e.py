# YOUR CODE HERE
N, Q = map(int, input().split())
queries = list(map(int, input().split()))

# Initialize sequence A and set S
A = [0] * N
S = set()

# Process each query
for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    # Update A for each element in S
    for j in S:
        A[j-1] += len(S)  # A is 1-indexed, so we use j-1 for 0-indexed array

# Print the final sequence
print(" ".join(map(str, A)))