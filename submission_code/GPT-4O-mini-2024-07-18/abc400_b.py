def calculate_sum(N, M):
    # Initialize the sum X
    X = 0
    current_term = 1  # This is N^0 which is 1
    
    for i in range(M + 1):
        X += current_term
        if X > 10**9:
            return "inf"
        current_term *= N  # Update current_term to N^i for the next iteration
    
    return X

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
M = int(data[1])

# Calculate and print the result
result = calculate_sum(N, M)
print(result)