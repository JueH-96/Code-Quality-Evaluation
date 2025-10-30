def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    R = input_data[1]
    C = input_data[2]

    # We'll create an N x N grid of '.' initially.
    grid = [['.' for _ in range(N)] for __ in range(N)]

    # used_in_col[j] will be a set of characters that are already used in column j.
    used_in_col = [set() for _ in range(N)]

    # We'll do a backtracking approach, row by row.
    # For row i, we must place exactly one A, one B, and one C in distinct columns.
    # The leftmost of these placed characters for row i must be R[i].
    # Additionally, for each column j, the first (topmost) character used must be C[j].
    
    from itertools import combinations, permutations

    # Attempt to fill row i. Return True if successful, else False.
    def backtrack(i):
        if i == N:
            # All rows assigned successfully
            return True
        
        # We must place A, B, C in row i, with the leftmost among them = R[i].
        row_chars = ['A', 'B', 'C']
        must_leftmost = R[i]
        # The other two characters in the row (besides must_leftmost).
        others = [ch for ch in row_chars if ch != must_leftmost]

        # Choose 3 columns out of N to place the A,B,C. The smallest of them must hold R[i].
        for cols in combinations(range(N), 3):
            sorted_cols = sorted(cols)
            # We enforce that the smallest of these columns is for R[i].
            leftmost_col = sorted_cols[0]

            # Check if we can place R[i] in leftmost_col.
            # Condition 1: R[i] not used in that column yet
            # Condition 2: If it's the first usage in that column, it must match C[leftmost_col].
            if R[i] in used_in_col[leftmost_col]:
                continue
            if not used_in_col[leftmost_col] and R[i] != C[leftmost_col]:
                # If this column is empty, its topmost must be C[col].
                continue

            # For the other two columns, we must place the two other characters in some permutation.
            for perm in permutations(others):
                # perm[0] goes to sorted_cols[1], perm[1] goes to sorted_cols[2]
                c1, c2 = perm[0], perm[1]
                col1, col2 = sorted_cols[1], sorted_cols[2]

                # Check column constraints for c1, col1
                if c1 in used_in_col[col1]:
                    continue
                if not used_in_col[col1] and c1 != C[col1]:
                    continue
                
                # Check column constraints for c2, col2
                if c2 in used_in_col[col2]:
                    continue
                if not used_in_col[col2] and c2 != C[col2]:
                    continue

                # If all checks pass, we tentatively place them.
                # Place R[i] in row i col leftmost_col
                grid[i][leftmost_col] = R[i]
                used_in_col[leftmost_col].add(R[i])

                # Place c1 in row i col col1
                grid[i][col1] = c1
                used_in_col[col1].add(c1)

                # Place c2 in row i col col2
                grid[i][col2] = c2
                used_in_col[col2].add(c2)

                # Backtrack for the next row
                if backtrack(i + 1):
                    return True

                # Revert
                grid[i][leftmost_col] = '.'
                used_in_col[leftmost_col].remove(R[i])

                grid[i][col1] = '.'
                used_in_col[col1].remove(c1)

                grid[i][col2] = '.'
                used_in_col[col2].remove(c2)
        
        return False

    # Run backtracking from row 0
    if backtrack(0):
        # If successful, print the grid
        print("Yes")
        for i in range(N):
            print("".join(grid[i]))
    else:
        print("No")

# Call main
if __name__ == "__main__":
    main()