def main():
    import sys, itertools
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    N = int(data[0])
    R = data[1].strip()
    C = data[2].strip()

    # The grid will be represented as a list of lists.
    grid = [['.' for _ in range(N)] for _ in range(N)]
    # For each column, we store the set of letters already used in that column.
    col_sets = [set() for _ in range(N)]
    # For each column, we record whether its top letter has already been set.
    # When a column is used for the first time (i.e. in the earliest row that uses it),
    # the letter placed must equal the corresponding letter in C.
    col_top = [False] * N

    letters = ['A', 'B', 'C']

    # Backtracking function: row -> bool
    def backtrack(row):
        # When we have assigned all rows, check that every column has exactly three letters.
        if row == N:
            for j in range(N):
                if len(col_sets[j]) != 3:
                    return False
            return True

        # Prune: For each column, even if we fill one letter per remaining row, we must end up with 3 letters.
        remaining = N - row  # number of rows left (each can contribute at most one letter for a column)
        for j in range(N):
            # We need exactly 3 letters per column. If already assigned count is cnt,
            # then the remaining missing letters is (3 - cnt). If that is greater than the rows left, no solution.
            if (3 - len(col_sets[j])) > remaining:
                return False

        # For the current row, we want to choose exactly 3 distinct columns to place letters.
        # We'll iterate over all combinations of 3 columns (sorted in increasing order).
        for comb in itertools.combinations(range(N), 3):
            # For each chosen combination, we need to assign a permutation of {A, B, C}
            # to these three positions, with the requirement that the set of letters in the row is exactly {A,B,C}.
            for perm in itertools.permutations(letters):
                # The leftmost filled cell (i.e. at the smallest column in the combination)
                # must have the letter matching R for this row.
                if perm[0] != R[row]:
                    continue

                valid = True
                # For each chosen cell (row, col) corresponding to comb[k] with letter perm[k],
                # if the column has not been used yet (col_top == False) then the letter must equal C[j],
                # because that cell will be the topmost in that column.
                for k, j in enumerate(comb):
                    if not col_top[j] and perm[k] != C[j]:
                        valid = False
                        break
                    # Also, this column already may have some letters; we cannot duplicate a letter in a given column
                    if perm[k] in col_sets[j]:
                        valid = False
                        break
                if not valid:
                    continue

                # Everything ok so far for the row.
                # Save the current state for these columns so we can backtrack.
                old_col_top = {}
                old_col_sets = {}
                for j in comb:
                    old_col_top[j] = col_top[j]
                    old_col_sets[j] = set(col_sets[j])

                # Place the letters in the current row.
                for k, j in enumerate(comb):
                    grid[row][j] = perm[k]
                    # If this column has not been used yet, now mark it as having its top letter assigned.
                    if not col_top[j]:
                        col_top[j] = True
                    col_sets[j].add(perm[k])
                
                # Recurse for the next row.
                if backtrack(row + 1):
                    return True
                # Backtrack: undo the assignments for the columns in comb.
                for j in comb:
                    grid[row][j] = '.'
                    col_top[j] = old_col_top[j]
                    col_sets[j] = old_col_sets[j]
        return False

    if backtrack(0):
        output_lines = ["Yes"]
        for row in grid:
            output_lines.append("".join(row))
        sys.stdout.write("
".join(output_lines))
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()