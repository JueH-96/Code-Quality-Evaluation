# YOUR CODE HERE
def min_operations(N, M, X, A):
    stones = [0] * (N + 1)
    total_stones = 0
    
    for i in range(M):
        stones[X[i]] = A[i]
        total_stones += A[i]
    
    if total_stones != N:
        return -1
    
    operations = 0
    excess = 0
    
    for i in range(1, N + 1):
        excess += stones[i] - 1
        operations += abs(excess)
    
    return operations

# Read input
N, M = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Solve and print output
result = min_operations(N, M, X, A)
print(result)