from itertools import permutations

def min_swaps_to_match(A, B):
    H, W = len(A), len(A[0])
    
    # Check if A can be transformed into B by swapping rows
    def can_transform_by_rows(A, B):
        for perm in permutations(range(H)):
            if all(A[perm[i]][j] == B[i][j] for i in range(H) for j in range(W)):
                return True, sum(i != perm[i] for i in range(H))
        return False, float('inf')
    
    # Check if A can be transformed into B by swapping columns
    def can_transform_by_columns(A, B):
        for perm in permutations(range(W)):
            if all(A[i][perm[j]] == B[i][j] for i in range(H) for j in range(W)):
                return True, sum(j != perm[j] for j in range(W))
        return False, float('inf')
    
    # Try transforming by rows and columns separately
    can_transform_rows, row_swaps = can_transform_by_rows(A, B)
    can_transform_cols, col_swaps = can_transform_by_columns(A, B)
    
    # If neither is possible, return -1
    if not can_transform_rows and not can_transform_cols:
        return -1
    
    # Return the minimum number of swaps needed
    return min(row_swaps, col_swaps)

# Read input
H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]

# Solve the problem
result = min_swaps_to_match(A, B)

# Write output
print(result)