def inversion_count(perm):
    # Count the minimum number of adjacent swaps needed to achieve a given permutation.
    # This is equal to the inversion count.
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    H = int(next(it))
    W = int(next(it))
    
    # Read grid A.
    A = []
    for _ in range(H):
        A.append([int(next(it)) for _ in range(W)])
        
    # Read grid B.
    B = []
    for _ in range(H):
        B.append([int(next(it)) for _ in range(W)])
    
    # Given the allowed operations we can arbitrarily permute rows
    # and columns by a series of adjacent swaps.
    # Thus any valid rearrangement is achievable.
    # For each permutation pair (row permutation and column permutation)
    # that transforms grid A into grid B, the cost is the sum of the
    # required numbers of adjacent swaps for rows (the inversion count
    # of the row permutation) plus that for columns.
    # We search over all possibilities (H, W <= 5 gives at most 14400 checks).
    
    from itertools import permutations
    best_cost = None
    rows_index = list(range(H))
    cols_index = list(range(W))
    
    for row_perm in permutations(rows_index):
        cost_rows = inversion_count(list(row_perm))
        for col_perm in permutations(cols_index):
            cost_cols = inversion_count(list(col_perm))
            valid = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                cost = cost_rows + cost_cols
                if best_cost is None or cost < best_cost:
                    best_cost = cost
    
    if best_cost is None:
        print(-1)
    else:
        print(best_cost)

if __name__ == "__main__":
    main()