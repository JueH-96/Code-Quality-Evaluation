def min_operations_to_transform(N, M, A, B):
    total_operations = 0
    
    for i in range(N):
        # Calculate the minimum operations to transform A[i] to B[i]
        diff = (B[i] - A[i]) % M
        reverse_diff = (A[i] - B[i]) % M
        min_operations = min(diff, reverse_diff)
        
        # Check if transforming A[i] to B[i] is possible without breaking the good sequence
        if i > 0 and B[i] == B[i-1]:
            return -1
        if i < N-1 and B[i] == B[i+1]:
            return -1
        
        total_operations += min_operations
    
    return total_operations

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))

# Get the result
result = min_operations_to_transform(N, M, A, B)

# Print the result
print(result)