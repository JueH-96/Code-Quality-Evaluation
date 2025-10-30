def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:1+N]))
    Q = list(map(int, data[1+N:1+2*N]))

    # ---------------------------------------------------------------------
    # Explanation of "X < Y in lexicographical order":
    #
    # For two strings X = X_1 X_2 ... X_|X| and Y = Y_1 Y_2 ... Y_|Y|,
    # we say "X < Y in lexicographical order" if EITHER:
    #
    #   1) |X| < |Y| AND X_1 X_2 ... X_|X| = Y_1 Y_2 ... Y_|X|
    #      (i.e., X is a proper prefix of Y), OR
    #
    #   2) There exists an integer i (1 <= i <= min(|X|,|Y|)) such that
    #      X_1 X_2 ... X_{i-1} = Y_1 Y_2 ... Y_{i-1}
    #      AND X_i < Y_i.
    #
    # Intuitively, the first place where they differ decides the comparison;
    # or if one is a prefix of the other, the shorter one is considered smaller.
    # ---------------------------------------------------------------------

    #
    # SOLUTION IDEA:
    #
    # Let p[i] be the "rank" of row i in permutation P.  That is, p[i] = k
    # where i is the k-th element in P's order (1-based).  Similarly, let
    # q[j] be the rank of column j in permutation Q (also 1-based).
    #
    # We fill the cell (i, j) with '1' if and only if p[i] + q[j] > N,
    # otherwise '0'.
    #
    # One can prove this construction satisfies:
    #   - Row ordering in the order P (i.e., S_{P_1} < S_{P_2} < ... < S_{P_N}),
    #   - Column ordering in the order Q (i.e., T_{Q_1} < T_{Q_2} < ... < T_{Q_N}),
    # both under the usual lexicographical comparison.
    #

    # Step 1: Build rank arrays p[] and q[] (each in range 1..N).
    #         p[i-1] = position of i in P
    #         q[j-1] = position of j in Q
    p = [0]*N
    for rank, row_idx in enumerate(P, start=1):
        p[row_idx-1] = rank

    q = [0]*N
    for rank, col_idx in enumerate(Q, start=1):
        q[col_idx-1] = rank

    # Step 2: Construct and output the N lines of the grid.
    #         A_{i,j} = '1' if p[i] + q[j] > N, else '0'.
    #
    # Here i and j are 0-based in the code, but conceptually 1..N in math.
    out = []
    for i in range(N):
        row_bits = []
        for j in range(N):
            if p[i] + q[j] > N:
                row_bits.append('1')
            else:
                row_bits.append('0')
        out.append("".join(row_bits))

    print("
".join(out))

# Call main() at the end
if __name__ == "__main__":
    main()