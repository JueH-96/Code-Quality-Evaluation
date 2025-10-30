import itertools

def count_inversions(perm):
    n = len(perm)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv_count += 1
    return inv_count

H, W = map(int, input().split())

A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]

# Check if multisets are the same
flat_A = sorted([x for row in A for x in row])
flat_B = sorted([x for row in B for x in row])
if flat_A != flat_B:
    print(-1)
    exit()

min_swaps = float('inf')

# Generate all possible row permutations
for r_perm in itertools.permutations(range(H)):
    # Apply row permutation to A
    A_row = [A[r] for r in r_perm]
    
    # Check if each row in A_row can be permuted to match B's rows
    valid = True
    for i in range(H):
        if sorted(A_row[i]) != sorted(B[i]):
            valid = False
            break
    if not valid:
        continue
    
    # Find all possible column permutations for the first row
    possible_c_perms = []
    for c_perm in itertools.permutations(range(W)):
        transformed = [A_row[0][c] for c in c_perm]
        if transformed == B[0]:
            possible_c_perms.append(c_perm)
    
    # Check each possible c_perm against all rows
    for c_perm in possible_c_perms:
        all_match = True
        for i in range(H):
            transformed_row = [A_row[i][c] for c in c_perm]
            if transformed_row != B[i]:
                all_match = False
                break
        if all_match:
            # Calculate inversion counts
            row_inv = count_inversions(r_perm)
            col_inv = count_inversions(c_perm)
            total = row_inv + col_inv
            if total < min_swaps:
                min_swaps = total

if min_swaps != float('inf'):
    print(min_swaps)
else:
    print(-1)