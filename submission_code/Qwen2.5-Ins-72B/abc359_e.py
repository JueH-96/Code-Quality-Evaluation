# YOUR CODE HERE
def solve(n, h):
    from itertools import accumulate
    
    # Initialize the array A with zeros
    A = [0] * (n + 1)
    
    # Initialize the result array to store the number of operations
    result = [0] * n
    
    # Initialize the operation counter
    operations = 0
    
    # Function to perform the operations and update A
    def perform_operations():
        nonlocal operations
        while True:
            operations += 1
            A[0] += 1
            for i in range(1, n + 1):
                if A[i - 1] > A[i] and A[i - 1] > h[i - 1]:
                    A[i - 1] -= 1
                    A[i] += 1
            if A[n] > 0:
                break
    
    # Perform the operations and record the number of operations for each A[i] > 0
    for i in range(1, n + 1):
        while A[i] == 0:
            perform_operations()
        result[i - 1] = operations
    
    return result

# Read input
n = int(input())
h = list(map(int, input().split()))

# Solve the problem
result = solve(n, h)

# Print the result
print(" ".join(map(str, result)))