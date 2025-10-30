def minimum_cost_to_make_identical(N, A, B, C):
    total_cost = 0
    
    # Calculate the initial cost of A
    current_cost = sum(A[i] * C[i] for i in range(N))
    
    # Iterate through each index to determine if a flip is needed
    for i in range(N):
        if A[i] != B[i]:  # A[i] needs to be flipped to match B[i]
            # Calculate the cost after flipping A[i]
            new_cost = current_cost - A[i] * C[i] + (1 - A[i]) * C[i]
            total_cost += new_cost
            current_cost = new_cost  # Update current cost to the new cost
            A[i] = 1 - A[i]  # Flip A[i]
    
    return total_cost

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
A = list(map(int, data[1].split()))
B = list(map(int, data[2].split()))
C = list(map(int, data[3].split()))

result = minimum_cost_to_make_identical(N, A, B, C)
print(result)