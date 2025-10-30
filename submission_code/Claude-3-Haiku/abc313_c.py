from collections import deque

N = int(input())
A = list(map(int, input().split()))

# Sort the array in ascending order
A.sort()

# Initialize the minimum and maximum values
min_val = A[0]
max_val = A[-1]

# Initialize the number of operations
operations = 0

# Perform the operations until the difference between the minimum and maximum values is at most 1
while max_val - min_val > 1:
    # Choose the indices i and j
    i = 0
    j = N - 1
    
    # Decrease A[i] by 1 and increase A[j] by 1
    A[i] -= 1
    A[j] += 1
    
    # Update the minimum and maximum values
    min_val = min(min_val, A[i])
    max_val = max(max_val, A[j])
    
    # Increment the number of operations
    operations += 1
    
    # Sort the array in ascending order
    A.sort()

print(operations)