def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:1+N]))
    Q = list(map(int, data[1+N:1+2*N]))
    
    # ------------------------------------------------------------
    # EXPLANATION OF THE CORE IDEA:
    #
    # We want:
    #   S_{P_1} < S_{P_2} < ... < S_{P_N}    (row-strings in lex order
    #                                        according to permutation P)
    # and
    #   T_{Q_1} < T_{Q_2} < ... < T_{Q_N}    (column-strings in lex order
    #                                        according to permutation Q)
    #
    # A neat construction that works is as follows:
    #
    #   1) Convert P into a "row-rank" array rpos[], so that
    #      rpos[i] = (0-based) index of i in P.
    #      i.e. if P_k = i, then rpos[i] = k-1 in 0-based indexing.
    #
    #   2) Convert Q into a "column-rank" array cpos[], so that
    #      cpos[j] = (0-based) index of j in Q.
    #
    #   3) Fill the cell (i, j) with '0' if rpos[i] + cpos[j] < N,
    #      and '1' otherwise.  (All i, j are 1-based, so adjust
    #      to 0-based carefully.)
    #
    # Intuition:
    #   - Rows corresponding to smaller rpos[i] will have more leading '0's
    #     and thus be lexicographically smaller.
    #   - Columns corresponding to smaller cpos[j] will similarly end up
    #     lexicographically smaller when read top-to-bottom.
    #
    # This can be shown to satisfy the required inequalities for
    # S and T under the permutations P and Q.
    #
    # ------------------------------------------------------------
    
    # Step 1: Build rpos[] so that rpos[x] = 0-based index of x in P
    # P is given 1-based, i.e. P = [P1, P2, ..., PN]
    # We'll store them 1-based internally for ease, then do 0-based rank.
    rpos = [0]*(N+1)  # rpos[i] will be the 0-based rank of i in P
    for rank, i in enumerate(P):
        rpos[i] = rank  # rank goes 0..N-1
    
    # Step 2: Build cpos[] so that cpos[x] = 0-based index of x in Q
    cpos = [0]*(N+1)
    for rank, j in enumerate(Q):
        cpos[j] = rank  # rank goes 0..N-1
    
    # Step 3: Construct the grid.
    # For each cell (i, j) with i,j in [1..N], define
    #   A_{i,j} = '0' if rpos[i] + cpos[j] < N, else '1'
    
    # We'll build each row as a list of characters and then join at the end.
    result = []
    for i in range(1, N+1):
        row_rank = rpos[i]
        row_chars = []
        for j in range(1, N+1):
            col_rank = cpos[j]
            if row_rank + col_rank < N:
                row_chars.append('0')
            else:
                row_chars.append('1')
        result.append("".join(row_chars))
    
    # Print the result
    print("
".join(result))

# Do not forget to call main().
main()