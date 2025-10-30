# Read input
N, K, Q = map(int, input().split())

# Initialize array A with zeros
A = [0] * N

# Process Q updates
for _ in range(Q):
    X, Y = map(int, input().split())
    # Update A (X is 1-indexed, so subtract 1)
    A[X-1] = Y
    
    # Compute f(A) - sum of K largest elements
    # Sort in descending order and take first K elements
    f_A = sum(sorted(A, reverse=True)[:K])
    
    print(f_A)