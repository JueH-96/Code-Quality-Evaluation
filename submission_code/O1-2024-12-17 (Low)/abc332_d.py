def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    
    # Read grid A
    idx = 2
    A = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        A.append(row)
        idx += W

    # Read grid B
    B = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        B.append(row)
        idx += W

    # If A already equals B, answer is 0
    if A == B:
        print(0)
        return

    # Precompute permutations of row/column indices
    # We'll use itertools.permutations for all permutations.
    # Then we will check for each (row_perm, col_perm) if
    # applying that permutation of rows and columns to A equals B.
    # The cost is the sum of the minimal adjacent-swaps cost for row_perm and col_perm.
    import itertools

    def count_inversions(perm):
        # Count the number of inversions in the permutation
        # which is the same as the minimal number of adjacent swaps needed.
        inv = 0
        n = len(perm)
        # Simple O(n^2) is fine for n <= 5
        for i in range(n):
            for j in range(i+1, n):
                if perm[i] > perm[j]:
                    inv += 1
        return inv

    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))

    # Convert A and B to a list of lists for easier indexing, 
    # though we already have them as such.

    # We'll store a dictionary from each row_perm to its cost_of_adj_swaps
    row_cost = {}
    for rp in row_perms:
        row_cost[rp] = count_inversions(rp)

    # Similarly for column perms
    col_cost = {}
    for cp in col_perms:
        col_cost[cp] = count_inversions(cp)

    # Check all permutations
    INF = float('inf')
    ans = INF
    for rp in row_perms:
        for cp in col_perms:
            # Check if for all i, j, A[i][j] == B[rp[i]][cp[j]]
            # If not match, skip
            match = True
            for i in range(H):
                if not match:
                    break
                for j in range(W):
                    if A[i][j] != B[rp[i]][cp[j]]:
                        match = False
                        break
            if match:
                cost = row_cost[rp] + col_cost[cp]
                if cost < ans:
                    ans = cost

    if ans == INF:
        print(-1)
    else:
        print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()