# YOUR CODE HERE
def find_triple(A, X, N):
    # A is 0-indexed, but we'll return 1-indexed
    seen = {A[0]: 0}  # Initialize with the first element
    for j in range(1, N-1):  # j ranges from 1 to N-2
        for k in range(j+1, N):  # k ranges from j+1 to N-1
            complement = X - A[j] - A[k]
            if complement in seen and seen[complement] < j:  # Ensure the index is less than j
                # Return 1-indexed indices
                return seen[complement]+1, j+1, k+1
        seen[A[j]] = j  # Add A[j] to the hash map after checking
    return -1

# Read input
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Find triple
result = find_triple(A, X, N)

# Print output
if result == -1:
    print(-1)
else:
    print(*result)