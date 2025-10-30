import itertools

H, W = map(int, input().split())

A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]

min_ops = None

# Generate all possible row permutations
for row_perm in itertools.permutations(range(H)):
    # Generate all possible column permutations
    for col_perm in itertools.permutations(range(W)):
        # Apply row permutation to A
        transformed = []
        for r in row_perm:
            row = A[r]
            # Apply column permutation to the row
            new_row = [row[c] for c in col_perm]
            transformed.append(new_row)
        # Check if transformed matches B
        if transformed != B:
            continue
        
        # Calculate inversion count for row permutation
        row_inv = 0
        for i in range(H):
            for j in range(i + 1, H):
                if row_perm[i] > row_perm[j]:
                    row_inv += 1
        
        # Calculate inversion count for column permutation
        col_inv = 0
        for i in range(W):
            for j in range(i + 1, W):
                if col_perm[i] > col_perm[j]:
                    col_inv += 1
        
        total = row_inv + col_inv
        if min_ops is None or total < min_ops:
            min_ops = total

print(min_ops if min_ops is not None else -1)