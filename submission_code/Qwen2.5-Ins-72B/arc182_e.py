# YOUR CODE HERE
import sys

def solve(N, M, C, K, A):
    # Initialize the result
    result = 0
    
    # Create a list to store the minimum values for each k
    min_values = [float('inf')] * K
    
    # Precompute the values for each A_i
    for i in range(N):
        current = A[i]
        for k in range(K):
            min_values[k] = min(min_values[k], current)
            current = (current + C) % M
    
    # Sum the minimum values
    result = sum(min_values)
    
    return result

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
C = int(data[2])
K = int(data[3])
A = list(map(int, data[4:]))

# Solve and print the result
print(solve(N, M, C, K, A))