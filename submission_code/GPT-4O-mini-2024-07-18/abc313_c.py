def min_operations_to_balance(N, A):
    max_A = max(A)
    min_A = min(A)
    
    # If the difference is already at most 1, no operations are needed
    if max_A - min_A <= 1:
        return 0
    
    # Calculate the total number of operations needed
    total_operations = 0
    for value in A:
        if value < max_A - 1:
            total_operations += (max_A - 1) - value
    
    return total_operations

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

result = min_operations_to_balance(N, A)
print(result)