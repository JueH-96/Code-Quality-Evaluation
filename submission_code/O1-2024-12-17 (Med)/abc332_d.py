def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    
    # Read grid A
    idx = 2
    A = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        idx += W
        A.append(row)
    
    # Read grid B
    B = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        idx += W
        B.append(row)

    from math import inf
    from itertools import permutations

    # Function to count the number of inversions in a permutation
    # (i.e., the minimum number of adjacent swaps needed to get from the identity to that permutation).
    def count_inversions(perm):
        inv_count = 0
        n = len(perm)
        # A simple O(n^2) method is fine for n <= 5
        for i in range(n):
            for j in range(i+1, n):
                if perm[i] > perm[j]:
                    inv_count += 1
        return inv_count

    # Check if applying row_perm to rows and col_perm to columns of A produces B.
    def match(A, B, row_perm, col_perm):
        for i in range(H):
            for j in range(W):
                if A[row_perm[i]][col_perm[j]] != B[i][j]:
                    return False
        return True

    # Generate all permutations of row indices and column indices
    row_indices = list(range(H))
    col_indices = list(range(W))
    min_ops = inf
    
    for row_perm in permutations(row_indices):
        # Count cost of row_perm
        row_cost = count_inversions(row_perm)
        for col_perm in permutations(col_indices):
            # Count cost of col_perm
            col_cost = count_inversions(col_perm)
            # Check if we get B
            if match(A, B, row_perm, col_perm):
                total_cost = row_cost + col_cost
                if total_cost < min_ops:
                    min_ops = total_cost

    if min_ops == inf:
        print(-1)
    else:
        print(min_ops)

# Do not forget to call main()
if __name__ == "__main__":
    main()