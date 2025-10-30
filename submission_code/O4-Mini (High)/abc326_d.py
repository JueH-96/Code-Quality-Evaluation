def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    R = data[1].strip()
    C = data[2].strip()
    # map A,B,C <-> 0,1,2
    char2idx = {'A':0, 'B':1, 'C':2}
    idx2char = ['A','B','C']
    R_idx = [char2idx[ch] for ch in R]
    C_idx = [char2idx[ch] for ch in C]

    # col_symbol_count[j][s] == 1 if column j already has symbol s
    col_symbol_count = [[0]*3 for _ in range(N)]
    # col_counts[j] = how many symbols placed so far in column j
    col_counts = [0]*N
    # whether column j has been filled at least once
    col_filled = [False]*N

    # row_assignments[i][j] = symbol at (i,j) (0,1,2) or -1 if empty
    row_assignments = [[-1]*N for _ in range(N)]

    solution = []

    def dfs(i):
        # i = current row to fill
        # prune: every column j needs exactly 3 symbols total
        rem_rows = N - i
        for j in range(N):
            if col_counts[j] + rem_rows < 3:
                return False
        if i == N:
            # check all columns got exactly 3
            if all(col_counts[j] == 3 for j in range(N)):
                # record solution
                for r in range(N):
                    solution.append(row_assignments[r][:])
                return True
            return False

        sMin = R_idx[i]
        # build candidate columns for each symbol
        colCands = [[] for _ in range(3)]
        for s in (0,1,2):
            for j in range(N):
                if col_symbol_count[j][s] != 0: 
                    continue
                if col_counts[j] >= 3:
                    continue
                # if this column is empty so far, the first symbol must match C[j]
                if not col_filled[j] and C_idx[j] != s:
                    continue
                colCands[s].append(j)

        # the symbol at the leftmost of the three must be R[i] == sMin
        others = [x for x in (0,1,2) if x != sMin]
        for jMin in colCands[sMin]:
            # the other two columns must be > jMin to make jMin leftmost
            L1 = [j for j in colCands[others[0]] if j > jMin]
            if not L1:
                continue
            L2 = [j for j in colCands[others[1]] if j > jMin]
            if not L2:
                continue
            for j1 in L1:
                for j2 in L2:
                    if j1 == j2:
                        continue
                    # place sMin at jMin, others[0] at j1, others[1] at j2
                    to_assign = [
                        (jMin,    sMin),
                        (j1,      others[0]),
                        (j2,      others[1])
                    ]
                    # apply
                    for j, s in to_assign:
                        row_assignments[i][j] = s
                        col_symbol_count[j][s] = 1
                        col_counts[j] += 1
                        if col_counts[j] == 1:
                            col_filled[j] = True
                    # recurse
                    if dfs(i+1):
                        return True
                    # undo
                    for j, s in to_assign:
                        row_assignments[i][j] = -1
                        col_symbol_count[j][s] = 0
                        col_counts[j] -= 1
                        if col_counts[j] == 0:
                            col_filled[j] = False
        return False

    if dfs(0):
        # print the found solution
        out = ["Yes"]
        for r in range(N):
            row = []
            for j in range(N):
                if solution[r][j] == -1:
                    row.append('.')
                else:
                    row.append(idx2char[solution[r][j]])
            out.append("".join(row))
        sys.stdout.write("
".join(out))
    else:
        sys.stdout.write("No
")

if __name__ == "__main__":
    main()