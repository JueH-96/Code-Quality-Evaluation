import sys

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sort the sequence A
A.sort()

# Initialize the minimum difference
min_diff = float('inf')

# Try all possible combinations of removing K elements
for i in range(N-K+1):
    # Remove the K elements from the current position
    B = A[:i] + A[i+K:]
    
    # Calculate the maximum and minimum values of B
    max_B = max(B)
    min_B = min(B)
    
    # Update the minimum difference if necessary
    min_diff = min(min_diff, max_B - min_B)

# Print the answer
print(min_diff)