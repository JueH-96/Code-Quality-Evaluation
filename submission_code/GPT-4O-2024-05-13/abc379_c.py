# YOUR CODE HERE
def min_operations_to_distribute_stones(N, M, X, A):
    total_stones = sum(A)
    
    # If total stones are not equal to N, it's impossible to distribute them evenly
    if total_stones != N:
        return -1
    
    # Create an array to represent the cells
    cells = [0] * N
    for i in range(M):
        cells[X[i] - 1] = A[i]
    
    operations = 0
    for i in range(N - 1):
        if cells[i] > 1:
            excess = cells[i] - 1
            cells[i] -= excess
            cells[i + 1] += excess
            operations += excess
    
    return operations

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X = list(map(int, data[2:2 + M]))
A = list(map(int, data[2 + M:]))

print(min_operations_to_distribute_stones(N, M, X, A))