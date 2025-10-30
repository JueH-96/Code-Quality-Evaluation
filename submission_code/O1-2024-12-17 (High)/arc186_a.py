def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    Q = int(data[1])
    queries = list(map(int, data[2:]))

    # We will precompute all possible numbers of "fixed" cells that can arise
    # in an N×N 0/1-matrix under the given definition.
    #
    # -------------------------------------------------------------------------
    # Explanation of the Key Idea
    # -------------------------------------------------------------------------
    # Recall the problem's definitions:
    #
    #  • Two matrices A and B (both N×N with 0/1-entries) are "similar" if
    #    they have the same row-sums and the same column-sums.
    #
    #  • A cell (i,j) in A is "fixed" if, for every B similar to A, the entry
    #    B[i,j] is the same as A[i,j].  Equivalently, you cannot flip that
    #    cell from 0→1 or 1→0 without violating the row/column-sum constraints
    #    that define the similarity class.
    #
    # We want to know, for each K in the queries, whether there exists an
    # N×N 0/1-matrix A such that exactly K cells are fixed (under the above
    # notion of "similar").
    #
    # -------------------------------------------------------------------------
    # High-Level Approach
    # -------------------------------------------------------------------------
    # One can show (with a more detailed combinatorial / flow argument) that
    # to force certain rows or columns completely to 0's or to 1's is the main
    # mechanism by which cells become "fixed."  Meanwhile, any "leftover"
    # submatrix (that does not have its row-sums = 0 or N, nor column-sums = 0
    # or N) can usually be rearranged in more than one way while preserving
    # the row and column sums (using standard "rectangle swaps" or similar in
    # bipartite graphs).  That leftover portion therefore contributes no fixed
    # cells.
    #
    # Concretely, we can think:
    #  • A row i with row-sum = N is forced to be all 1's (those N cells
    #    are all fixed to 1).
    #  • A row i with row-sum = 0 is forced to be all 0's (those N cells
    #    are all fixed to 0).
    #  • Similarly for columns with sum = N or sum = 0.
    #
    # After removing such "fully-saturated" or "fully-empty" rows/columns,
    # one obtains a smaller leftover submatrix whose row-/col-sums lie
    # strictly between 0 and N.  In most cases (as soon as it's at least
    # 1×2, 2×1, or bigger), that leftover submatrix can be rearranged in
    # at least two different ways, so none of its cells are fixed.
    #
    # Thus the number of fixed cells is exactly those coming from the union
    # of rows forced entirely to 0 or 1, plus columns forced entirely to 0 or 1.
    # One can systematically count how many are forced that way for all
    # possible patterns of fully-0 / fully-N row-sums and col-sums.  We also
    # make sure there is a leftover submatrix that still admits at least two
    # solutions (else the entire matrix is unique and all N^2 cells end up
    # forced).  Or we can have a leftover dimension = 0 or 1×1, which yields
    # a unique matrix and hence N^2 forced cells.
    #
    # It turns out one can parameterize all such possibilities by picking:
    #
    #    r = number of rows that are either sum=0 or sum=N
    #    c = number of columns that are either sum=0 or sum=N
    #    then, among those r rows, choose how many have sum=N (call it 'a')
    #    the rest have sum=0  =>  x = r - a
    #    among those c columns, choose how many have sum=N (call it 'b')
    #    the rest have sum=0  =>  y = c - b
    #
    # The size of the leftover submatrix is (N-r) × (N-c).  If that leftover
    # dimension is 0 (meaning r=N or c=N) or is 1×1, the matrix is forced
    # uniquely => that yields N^2 fixed.  Otherwise, that leftover submatrix
    # can be shown to have at least two distinct fillings => so only the
    # "saturated" rows/columns produce forced cells.  One can compute the
    # count of these forced cells by:
    #
    #    forced =   [a rows of all-1 => a*N cells]
    #             + [x rows of all-0 => x*N cells]
    #             + [b columns of all-1 => b*N cells]
    #             + [y columns of all-0 => y*N cells]
    #    but we subtract double-counted intersections.  It turns out if
    #    "a" rows are all-1 and "b" columns are all-1, you double-counted
    #    those a*b intersections, etc.  An equivalent formula is
    #
    #        forced = N*(a + x + b + y) - (a*b + x*y),
    #
    #    where a+b+x+y = r + c.  We then vary r,c in [0..N], a in [0..r],
    #    b in [0..c], etc.
    #
    # Rather than re-deriving each piece, a simpler direct code enumerates:
    #
    #    • r in [0..N], c in [0..N].
    #      leftover dimension = (N-r)×(N-c).
    #    • If leftover dimension is 0 or 1×1, forced = N^2.
    #      Otherwise, for each way a in [0..r], x=r-a, b in [0..c], y=c-b:
    #        forced_count = N*(r+c) - [ (a*b) + (x*y ) ].
    #
    # We'll collect all possible forced_count values in a set (bounded by 0..N^2).
    # Then for each query K, we answer "Yes" if K is in that set, else "No."
    #
    # The overall complexity is O(N^4), which is fine for N up to 30 (~1 million
    # iterations).
    #
    # -------------------------------------------------------------------------
    # Implementation
    # -------------------------------------------------------------------------

    # Boolean array for which fixed-counts are achievable:
    can_fixed = [False] * (N*N + 1)

    # Enumerate r, c = how many rows, columns are fully-0-or-N:
    # leftover dimension = (N-r) by (N-c).
    for r in range(N+1):
        for c in range(N+1):
            L = N - r  # leftover rows
            M = N - c  # leftover cols

            # If the leftover dimension is 0 in at least one direction,
            # or exactly 1×1, this forces a unique arrangement => N^2 fixed.
            if L == 0 or M == 0 or (L == 1 and M == 1):
                can_fixed[N*N] = True
                continue

            # Otherwise, there's a leftover region that can admit multiple
            # solutions.  So only the "fully-0-or-N" rows/cols produce forced cells.
            # We'll split r rows into a of them sum=N and x=r-a sum=0,
            # and c columns into b of them sum=N and y=c-b sum=0.
            # Then forced = N*(r+c) - [ a*b + (r-a)*(c-b) ].
            # We'll enumerate all a,b.
            for a in range(r+1):
                x = r - a
                for b in range(c+1):
                    y = c - b
                    val = N*(r + c) - (a*b + x*y)
                    if 0 <= val <= N*N:
                        can_fixed[val] = True

    # Now answer each query K.
    out = []
    for k in queries:
        if 0 <= k <= N*N and can_fixed[k]:
            out.append("Yes")
        else:
            out.append("No")

    print("
".join(out))


# Don't forget to call main().
if __name__ == "__main__":
    main()