import sys
from itertools import combinations, permutations

# -------------------------------------------------------------

def main() -> None:
    sys.setrecursionlimit(10000)

    # ---------------- read ----------------
    N   = int(sys.stdin.readline().strip())
    R   = sys.stdin.readline().strip()       # required left-most letter of each row
    C   = sys.stdin.readline().strip()       # required top-most letter of each column

    # ---------------- constants / helpers ----------------
    letters          = ['A', 'B', 'C']
    letter_bit       = {'A': 1, 'B': 2, 'C': 4}        # single bits 001 / 010 / 100
    idx_of_letter    = {'A': 0, 'B': 1, 'C': 2}

    # ---------------- pre-compute all legal patterns for every row ----------------
    # every pattern is a triple (col_of_A , col_of_B , col_of_C)
    row_patterns: list[list[tuple[int,int,int]]] = []
    for i in range(N):
        want_left = R[i]
        patterns  = []
        for cols in combinations(range(N), 3):
            for perm in permutations(cols):            # perm[0]→A , perm[1]→B , perm[2]→C
                leftmost_col = min(perm)
                # which letter is at that leftmost column?
                if   perm[0] == leftmost_col: left_letter = 'A'
                elif perm[1] == leftmost_col: left_letter = 'B'
                else:                         left_letter = 'C'
                if left_letter != want_left:
                    continue
                patterns.append(perm)                  # perm is (cA,cB,cC)
        row_patterns.append(patterns)

    # early impossible
    if any(len(p)==0 for p in row_patterns):
        print("No")
        return

    # ---------------- state containers ----------------
    grid           = [['.']*N for _ in range(N)]

    col_mask       = [0]*N             # bits already used in each column
    col_filled     = [0]*N             # how many letters already placed in column (0..3)
    top_letter     = ['']*N            # first letter already written in each column ('' = none yet)

    # ---------------- recursion ----------------
    def dfs(row: int) -> bool:
        if row == N:                                   # all rows processed
            return all(col_mask[j] == 7 and col_filled[j] == 3 for j in range(N))

        for pattern in row_patterns[row]:
            # ---------- check feasibility ----------
            ok = True
            for k in range(3):                         # k=0:A ,1:B ,2:C
                j      = pattern[k]
                letter = letters[k]
                if col_mask[j] & letter_bit[letter]:      # duplicate in column
                    ok = False; break
                if col_filled[j] >= 3:                     # column already full
                    ok = False; break
                if top_letter[j] == '' and letter != C[j]: # violates top-most requirement
                    ok = False; break
            if not ok:
                continue

            # ---------- apply this pattern ----------
            # keep history for easy rollback
            history = []
            for k in range(3):
                j      = pattern[k]
                letter = letters[k]
                history.append( (j, col_mask[j], col_filled[j], top_letter[j]) )

                col_mask[j]   |= letter_bit[letter]
                col_filled[j] += 1
                if top_letter[j] == '':
                    top_letter[j] = letter
                grid[row][j]   = letter

            # quick capacity pruning: each remaining row can add at most one letter to a column
            remain = N - row - 1
            if all( (3 - col_filled[j]) <= remain for j in range(N) ):
                if dfs(row + 1):
                    return True

            # ---------- rollback ----------
            for j, prev_mask, prev_fill, prev_top in history:
                col_mask[j]   = prev_mask
                col_filled[j] = prev_fill
                top_letter[j] = prev_top
                grid[row][j]  = '.'

        return False

    # ---------------- run search & output ----------------
    if dfs(0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

# -------------------------------------------------------------
if __name__ == "__main__":
    main()