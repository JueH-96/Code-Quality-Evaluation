import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    from itertools import combinations, permutations

    N = int(sys.stdin.readline().strip())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # The grid we will build; '.' means empty.
    grid = [['.' for _ in range(N)] for _ in range(N)]
    # For each column j, how many of each letter we have placed.
    col_counts = [ {'A':0, 'B':0, 'C':0} for _ in range(N) ]
    # For each column j, how many total filled cells.
    col_filled = [0]*N
    # For each column j, the row index of the topmost placed letter (or None).
    col_top_row = [None]*N
    # And what letter that topmost is.
    col_top_letter = [None]*N

    solution = None

    # Backtrack row by row.
    def backtrack(row):
        nonlocal solution
        if solution is not None:
            return

        if row == N:
            # All rows done: check final column constraints.
            # Each column must have exactly one of each A,B,C and topmost letter equals C[j].
            for j in range(N):
                if col_counts[j]['A']!=1 or col_counts[j]['B']!=1 or col_counts[j]['C']!=1:
                    return
                if col_top_letter[j] != C[j]:
                    return
            # Found valid
            sol = [''.join(grid[i]) for i in range(N)]
            solution = sol
            return

        # We must place exactly one A,B,C in this row, choose 3 distinct columns.
        for cols in combinations(range(N), 3):
            # For each permutation of letters in these three spots.
            for perm in permutations(['A','B','C']):
                # Check leftmost written matches R[row].
                min_col = min(cols)
                idx = cols.index(min_col)
                if perm[idx] != R[row]:
                    continue

                # Try to place; but do pruning on column-top constraints.
                ok = True
                updates = []
                for c_idx, col in enumerate(cols):
                    letter = perm[c_idx]
                    # Can't exceed column counts
                    if col_counts[col][letter] != 0:
                        ok = False
                        break
                    # Column total filled <=3
                    if col_filled[col] >= 3:
                        ok = False
                        break
                    # If column already has a topmost in a previous row:
                    tr = col_top_row[col]
                    if tr is not None and tr < row:
                        # That topmost is fixed; it must match the final C[col]
                        if col_top_letter[col] != C[col]:
                            ok = False
                            break
                    # If no topmost yet, this placement is the new topmost
                    if tr is None:
                        # Then this letter must match C[col], otherwise it's fixed wrong.
                        if letter != C[col]:
                            ok = False
                            break
                    updates.append((col, letter, tr is None))

                if not ok:
                    continue

                # Commit placements
                for (col, letter, is_new_top) in updates:
                    grid[row][col] = letter
                    col_counts[col][letter] += 1
                    col_filled[col] += 1
                    if col_top_row[col] is None:
                        col_top_row[col] = row
                        col_top_letter[col] = letter

                # Recurse
                backtrack(row+1)
                if solution is not None:
                    return

                # Undo
                for (col, letter, is_new_top) in updates:
                    grid[row][col] = '.'
                    col_counts[col][letter] -= 1
                    col_filled[col] -= 1
                    if is_new_top:
                        col_top_row[col] = None
                        col_top_letter[col] = None

    backtrack(0)

    if solution is None:
        print("No")
    else:
        print("Yes")
        for line in solution:
            print(line)

if __name__ == "__main__":
    main()