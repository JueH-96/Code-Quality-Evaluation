# YOUR CODE HERE
def find_operations(n, heights):
    # Initialize an array to store operations count for each A_i
    operations = [0] * n
    
    # A_0 must exceed H_1 for A_1 to become positive, which takes H_1 + 1 operations
    operations[0] = heights[0] + 1
    
    # For each subsequent A_i, simulate how many more operations are needed
    for i in range(1, n):
        # Initialize the state after the previous operation
        A = [0] * (n + 1)
        operation_count = operations[i - 1]
        
        # Simulate the state after the previous operation
        for j in range(operation_count):
            A[0] += 1
            for k in range(1, n + 1):
                if A[k-1] > A[k] and A[k-1] > heights[k-1]:
                    A[k-1] -= 1
                    A[k] += 1
        
        # Continue operations until A_i becomes positive
        while A[i] <= 0:
            operation_count += 1
            A[0] += 1
            for k in range(1, n + 1):
                if A[k-1] > A[k] and A[k-1] > heights[k-1]:
                    A[k-1] -= 1
                    A[k] += 1
        
        operations[i] = operation_count
    
    return operations

# Read input
n = int(input())
heights = list(map(int, input().split()))

# Compute operations
result = find_operations(n, heights)

# Output
print(" ".join(map(str, result)))