def is_geometric_progression(N, A):
    if N < 2:
        return "No"
    
    # Calculate the common ratio using the first two elements
    if A[0] == 0:
        return "No"  # To avoid division by zero, although A_i >= 1 by constraints
    
    common_ratio = A[1] / A[0]
    
    for i in range(1, N):
        if A[i] == 0 or A[i-1] == 0 or A[i] / A[i-1] != common_ratio:
            return "No"
    
    return "Yes"

import sys

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

# Check if the sequence is a geometric progression
result = is_geometric_progression(N, A)
print(result)