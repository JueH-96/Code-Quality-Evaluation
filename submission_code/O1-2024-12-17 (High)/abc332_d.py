def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    idx = 2
    A = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        idx += W
        A.append(row)
    B = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        idx += W
        B.append(row)
        
    from itertools import permutations
    
    # A helper function to count inversions in a permutation (the number of adjacent swaps needed).
    def count_inversions(arr):
        inv = 0
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]:
                    inv += 1
        return inv
    
    # We will try all permutations of row indices and column indices.
    # For each combination, we check if reordering A by those permutations matches B.
    # Then, if it matches, we calculate the cost = inversions_in_row_perm + inversions_in_col_perm
    # and keep track of the minimum cost.
    
    row_indices = list(range(H))
    col_indices = list(range(W))
    from math import inf
    min_operations = inf
    
    for row_perm in permutations(row_indices):
        for col_perm in permutations(col_indices):
            # Check if applying row_perm and col_perm to A matches B
            match = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                cost = count_inversions(row_perm) + count_inversions(col_perm)
                if cost < min_operations:
                    min_operations = cost

    if min_operations == inf:
        print(-1)
    else:
        print(min_operations)

# Do not forget to call main().
if __name__ == "__main__":
    main()