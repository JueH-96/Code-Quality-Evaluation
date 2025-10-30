def count_sequences(N, M, A):
    MOD = 998244353
    
    # Create a list to store the maximum value for each index
    max_values = [0] * N
    
    # Determine the maximum value that can be assigned to each position
    for i in range(N):
        max_values[i] = max(max_values[i], M)  # Start with the maximum possible value
        # For each A[i], we can propagate the maximum value
        if A[i] - 1 < N:
            max_values[A[i] - 1] = max(max_values[A[i] - 1], max_values[i])
    
    # Count the number of valid sequences
    result = 1
    for i in range(N):
        # Each position can take values from 1 to max_values[i]
        result = (result * max_values[i]) % MOD
    
    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))

# Calculate the result
result = count_sequences(N, M, A)

# Print the result
print(result)