# YOUR CODE HERE
def min_operations(N, S, T):
    # Check if transformation is possible
    if sorted(S) != sorted(T):
        return -1
    
    # Initialize the number of operations
    operations = 0
    
    # Convert strings to lists for easier manipulation
    S = list(S)
    T = list(T)
    
    # Two pointers approach
    i, j = 0, 0
    
    while i < N and j < N:
        # Find the next mismatch in S
        while i < N and S[i] == T[i]:
            i += 1
        # Find the next mismatch in T
        while j < N and S[j] == T[j]:
            j += 1
        
        if i < N and j < N:
            # Swap the mismatched stones
            S[i], S[j] = S[j], S[i]
            operations += 1
            i += 1
            j += 1
    
    return operations

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
T = data[2]

# Print the result
print(min_operations(N, S, T))