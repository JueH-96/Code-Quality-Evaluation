from itertools import permutations

def count_inversions(perm):
    """Count the minimum number of adjacent swaps needed to achieve this permutation"""
    count = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                count += 1
    return count

def apply_permutations(grid, row_perm, col_perm):
    """Apply row and column permutations to the grid"""
    H, W = len(grid), len(grid[0])
    result = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(grid[row_perm[i]][col_perm[j]])
        result.append(row)
    return result

# Read input
H, W = map(int, input().split())
A = []
for _ in range(H):
    row = list(map(int, input().split()))
    A.append(row)

B = []
for _ in range(H):
    row = list(map(int, input().split()))
    B.append(row)

min_ops = float('inf')

# Try all possible permutations of rows and columns
for row_perm in permutations(range(H)):
    for col_perm in permutations(range(W)):
        # Transform grid A using these permutations
        transformed = apply_permutations(A, row_perm, col_perm)
        
        # Check if the transformed grid matches B
        if transformed == B:
            # Calculate the number of operations needed
            row_ops = count_inversions(row_perm)
            col_ops = count_inversions(col_perm)
            total_ops = row_ops + col_ops
            min_ops = min(min_ops, total_ops)

# Output the result
if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)