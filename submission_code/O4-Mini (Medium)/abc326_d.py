def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    N = int(input().strip())
    R = input().strip()
    C = input().strip()

    # Map letters to indices 0,1,2
    ld = {'A': 0, 'B': 1, 'C': 2}
    # Precompute row options: for each row i, list of triples t=(colA,colB,colC)
    # where t[ld[R[i]]] is the minimum of the three entries
    row_options = []
    rng = range(N)
    for i in range(N):
        need = ld[R[i]]
        opts = []
        # Enumerate all distinct assignments of A,B,C to columns
        for a in rng:
            for b in rng:
                if b == a: continue
                for c in rng:
                    if c == a or c == b: continue
                    t = (a, b, c)
                    # Check leftmost letter constraint
                    if t[need] != min(t):
                        continue
                    opts.append(t)
        row_options.append(opts)

    # used_letter[j][l]: whether letter l (0..2) is used in column j
    used_letter = [[False]*3 for _ in range(N)]
    # col_non_empty_assigned[j]: whether any letter has been placed yet in column j
    col_non_empty_assigned = [False]*N

    sol = [None]*N  # to store chosen triple per row
    idxC = [ld[ch] for ch in C]

    def dfs(i):
        if i == N:
            # All rows assigned; by construction each row has 3 letters,
            # no duplicates per column per letter, and total counts force every
            # letter in each column exactly once.
            return True
        for t in row_options[i]:
            # Check if this row option t is consistent with column constraints
            ok = True
            # We'll need to record new columns that we mark non-empty now
            new_first = []
            for l_idx in range(3):
                j = t[l_idx]
                # cannot reuse same letter in a column
                if used_letter[j][l_idx]:
                    ok = False
                    break
                # if this column has no letters yet, the first letter must be C[j]
                if not col_non_empty_assigned[j] and l_idx != idxC[j]:
                    ok = False
                    break
            if not ok:
                continue
            # apply this choice
            for l_idx in range(3):
                j = t[l_idx]
                used_letter[j][l_idx] = True
                if not col_non_empty_assigned[j]:
                    col_non_empty_assigned[j] = True
                    new_first.append(j)
            sol[i] = t
            if dfs(i+1):
                return True
            # backtrack
            for l_idx in range(3):
                j = t[l_idx]
                used_letter[j][l_idx] = False
            for j in new_first:
                col_non_empty_assigned[j] = False
        return False

    if not dfs(0):
        print("No")
        return

    # Build the grid of characters
    grid = [ ['.']*N for _ in range(N) ]
    rev = ['A','B','C']
    for i in range(N):
        t = sol[i]
        for l_idx in range(3):
            j = t[l_idx]
            grid[i][j] = rev[l_idx]

    print("Yes")
    for row in grid:
        print("".join(row))


if __name__ == "__main__":
    main()