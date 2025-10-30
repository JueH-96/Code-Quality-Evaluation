# YOUR CODE HERE
from itertools import permutations

def min_operations_to_match_grids(H, W, A, B):
    def apply_permutation(grid, row_perm, col_perm):
        new_grid = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                new_grid[row_perm[i]][col_perm[j]] = grid[i][j]
        return new_grid

    row_indices = list(range(H))
    col_indices = list(range(W))
    
    min_operations = float('inf')
    found = False
    
    for row_perm in permutations(row_indices):
        for col_perm in permutations(col_indices):
            permuted_A = apply_permutation(A, row_perm, col_perm)
            if permuted_A == B:
                operations = sum(1 for i in range(H) if row_perm[i] != i) + sum(1 for j in range(W) if col_perm[j] != j)
                min_operations = min(min_operations, operations)
                found = True
    
    return min_operations if found else -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])

A = []
B = []

index = 2
for _ in range(H):
    A.append([int(data[index + j]) for j in range(W)])
    index += W

for _ in range(H):
    B.append([int(data[index + j]) for j in range(W)])
    index += W

# Calculate and print the result
result = min_operations_to_match_grids(H, W, A, B)
print(result)