def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    Q = [int(next(it)) for _ in range(N)]
    
    # We will use the following idea.
    # Define, for each row i (1-indexed), let R[i] be the “rank” of row i in the permutation P;
    # i.e. if P=[P1, P2, …, PN] then R[ P1 ] = 0, R[ P2 ] = 1, …, R[ PN ] = N-1.
    # Similarly, for each column j (1-indexed), let d[j] be the “rank” of column j in the permutation Q;
    # that is, if Q=[Q1, Q2, …, QN] then d[ Q1 ] = 0, d[ Q2 ] = 1, …, d[ QN ] = N-1.
    #
    # Then, define the N×N binary grid A by
    #   A[i][j] = '1' if and only if   R[i] + d[j] >= N,
    #            = '0' otherwise.
    #
    # Why does this work?
    # --------------------
    # (1) Row condition.
    # Let S_i be the i‑th row string in natural order of columns.
    # Consider two rows i and k with R[i] < R[k]. Look at column j where
    #   d[j] = N - 1 - R[i]  
    # (such a unique column exists since d is a permutation of 0,1,…,N–1).
    # Then for row i we have:
    #   R[i] + d[j] = R[i] + (N - 1 - R[i]) = N - 1, so A[i][j] = '0'
    # While for row k, since R[k] ≥ R[i]+1,
    #   R[k] + d[j] ≥ (R[i]+1) + (N - 1 - R[i]) = N, so A[k][j] = '1'.
    # In the lexicographical comparison (which scans columns in natural order from j=1 to N)
    # the first difference (at some column, and in particular at or later than the one we just found)
    # will be that S_i has a '0' and S_k a '1' – giving S_i < S_k.
    #
    # (2) Column condition.
    # Let T_j be the string in column j read top‐to‐bottom.
    # Now consider two columns j and k with d[j] < d[k]. We want T_j < T_k.
    # Look for a row i satisfying:
    #   R[i] < N - d[j]   and   R[i] ≥ N - d[k].
    # Because d[k] > d[j] the interval [N - d[k], N - d[j]) is nonempty
    # (its size is d[k] - d[j] ≥ 1). Since R is a permutation of 0,1,…,N–1,
    # there will be some row i with value in that interval.
    # For that row:
    #   For column j: R[i] + d[j] < (N - d[k) + d[j] ≤ N - 1  so A[i][j] = '0'.
    #   For column k: R[i] + d[k] ≥ (N - d[k) + d[k] = N so A[i][k] = '1'.
    # As before, in the lex‐comparison (scanning rows in natural order) the first difference
    # between T_j and T_k will be that T_j has '0' and T_k has '1'. Hence T_j < T_k.
    #
    # Since the rows (when re‐ordered by P) and the columns (when re‐ordered by Q) satisfy
    # strict lexicographical increasing order, the grid A is a valid answer.
    
    # Compute the 0-indexed rank for rows: R[ row ] for row in 1...N.
    R = [0] * (N + 1)
    for rank, r in enumerate(P):
        R[r] = rank  # so that R[P[0]]=0, R[P[1]]=1, ..., R[P[N-1]]=N-1.
    
    # Similarly, compute the rank for columns: d[ col ] for col in 1...N.
    d = [0] * (N + 1)
    for rank, col in enumerate(Q):
        d[col] = rank
    
    # Build the grid.
    # For i = 1,...,N and j = 1,...,N, set:
    #    A[i][j] = '1' if R[i] + d[j] >= N, else '0'.
    # (Recall: R and d are in the range 0 .. N-1.)
    result = []
    for i in range(1, N+1):
        row_chars = []
        Ri = R[i]
        for j in range(1, N+1):
            if Ri + d[j] >= N:
                row_chars.append('1')
            else:
                row_chars.append('0')
        result.append("".join(row_chars))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()