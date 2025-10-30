def solve():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # We want to fill an N×N grid with exactly one A, one B, one C per row and per column.
    # Additional constraints:
    # 1) The leftmost (non-dot) character in row i is R[i].
    # 2) The topmost (non-dot) character in column j is C[j].
    #
    # We will use backtracking over rows:
    # - For each row i, we choose exactly 3 distinct columns to place A,B,C.
    # - Among those chosen columns, the letter R[i] must go into the leftmost column index.
    # - The other two letters go into the other two chosen columns in either permutation.
    # - While we place letters in column j, if it's the first (topmost) letter in col j,
    #   then it must match C[j]. If we already placed a letter in a higher row in col j,
    #   then there's no additional restriction from topmost.
    # - We also track that each column can only have one of each letter (to ensure exactly
    #   one A, one B, one C in each column).
    # - If we manage to fill all rows, we then verify each column indeed has A,B,C exactly once.
    #
    # If a valid arrangement exists, print "Yes" and one such arrangement; otherwise print "No".

    # grid[r][c]: character in row r, column c ('.' if empty)
    grid = [['.' for _ in range(N)] for _ in range(N)]
    # colCount[j][letter] = how many times letter is used in column j
    colCount = [ {'A':0, 'B':0, 'C':0} for _ in range(N) ]
    # topmostUsed[j] = row index where the first (topmost) letter was placed in col j, or -1 if none
    topmostUsed = [-1]*N

    from itertools import combinations

    # letters we need in each row
    all_letters = ['A', 'B', 'C']

    def backtrack(row):
        if row == N:
            # Check if each column has exactly one A, B, C
            for j in range(N):
                if colCount[j]['A'] != 1 or colCount[j]['B'] != 1 or colCount[j]['C'] != 1:
                    return False
            return True

        # We must place exactly one A, B, C in this row.
        # The leftmost among the chosen columns must hold R[row].
        row_letter = R[row]
        other_letters = [x for x in all_letters if x != row_letter]

        # Try all combinations of 3 distinct columns for these letters
        # Let them be (x, y, z) with x < y < z. We require row_letter in x
        # because x is the leftmost among x,y,z.
        # Then the other two letters can go in y,z in either order.
        for cols in combinations(range(N), 3):
            x, y, z = sorted(cols)
            # We'll fix row_letter at column x
            # Then the other two letters from other_letters can go in (y,z)
            # in 2 permutations:
            from itertools import permutations
            for perm in permutations(other_letters):
                let_y, let_z = perm[0], perm[1]

                # We'll place row_letter in col x,
                # place let_y in col y,
                # place let_z in col z.

                # We check feasibility for each of these placements:
                # - colCount constraints (each letter used at most once in each column)
                # - topmost constraint (if topmostUsed for the column is -1,
                #   then the placed letter must be C[column])
                # If all are feasible, we do them and recurse.

                chosen_positions = [(x, row_letter), (y, let_y), (z, let_z)]

                feasible = True
                for cidx, letter in chosen_positions:
                    # Check if column cidx already used that letter
                    if colCount[cidx][letter] == 1:
                        feasible = False
                        break
                    # Check topmost constraint
                    if topmostUsed[cidx] == -1:
                        # Then the first letter in this column must match C[cidx]
                        if letter != C[cidx]:
                            feasible = False
                            break

                if not feasible:
                    continue

                # If feasible, do the assignment
                updates_topmost = []  # track columns for which we set topmostUsed
                for cidx, letter in chosen_positions:
                    grid[row][cidx] = letter
                    colCount[cidx][letter] += 1
                    if topmostUsed[cidx] == -1:
                        topmostUsed[cidx] = row
                        updates_topmost.append(cidx)

                # Recurse
                if backtrack(row+1):
                    return True

                # Otherwise revert the changes
                for cidx, letter in chosen_positions:
                    grid[row][cidx] = '.'
                    colCount[cidx][letter] -= 1
                for cidx in updates_topmost:
                    topmostUsed[cidx] = -1

        # If no combination works for this row, return False
        return False

    if backtrack(0):
        # We found a solution. Print it.
        print("Yes")
        for r in range(N):
            print("".join(grid[r]))
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()