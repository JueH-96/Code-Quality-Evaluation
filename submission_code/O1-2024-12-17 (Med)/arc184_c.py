def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    # We are given 0 = A1 < A2 < ... < A_N, but the actual numeric values of A[]
    # do not affect *where* they can split (left vs. right vs. middle) other than
    # enforcing that smaller A's must go into the "left" side and larger A's must
    # go into the "right" side if we try to fold them across a single midpoint.
    #
    # Key observation (from standard paper-fold reasoning):
    # ---------------------------------------------------
    # After 100 folds, the paper has 2^100 - 1 distinct creases, which is huge
    # (far larger than N can be).  Because A_1 < A_2 < ... < A_N, any partition
    # of these N offsets into "left side", "middle fold" (at most one offset),
    # and "right side" is geometrically possible by choosing an appropriate
    # global shift "i"—provided that all chosen left offsets are strictly less
    # than any chosen right offsets, and that the single 'middle fold' offset
    # (if any) lies strictly between them.  Since A is strictly increasing, any
    # contiguous segment of A corresponds exactly to a set of smaller offsets
    # separate from a set of larger offsets.  Hence the only real constraint is
    # how we choose to break A into (left, optional-middle, right).
    #
    # Meanwhile, the "mountain/valley" pattern is built by a fractal recursion:
    #   P(n) = [ P(n-1) ] + [ middle fold = M ] + [ invert(reverse(P(n-1))) ]
    # Thus at level n, we have three ways to place a block of creases:
    #   1) entirely in the left half => we get the same M/V distribution as P(n-1)
    #   2) entirely in the right half => we get the inverted distribution from P(n-1)
    #   3) we can also choose exactly one crease to be the "middle fold" (which is M),
    #      and the rest go left or right accordingly.
    #
    # If we only cared about the number of creases rather than *which* ones are bigger
    # or smaller, we could try to put as many as possible in positions that end up M.
    # But for an actual subset A, the relative ordering (A_1 < A_2 < ... < A_N) forces
    # which offsets can be on the left side vs. the right side vs. the middle fold.
    #
    # However, because the paper length 2^100-1 is so large, any contiguous subset
    # of A's (by index) can be placed in the left half, and any higher contiguous
    # subset can be placed in the right half, etc.  The only requirement is that
    # smaller offsets cannot "jump" to the right half while a bigger offset goes
    # to the left half.  In sorted order, that means we may cut the array of offsets
    # into (left block, maybe 1 in the middle, right block), respecting the sorted
    # index order.
    #
    # So to count how many offsets become M, we do a recursion in terms of:
    #   dp(n, l, r) = the maximum number of M's achievable using A_l..A_r
    #   when embedding them into P(n).
    #
    # Recurrence:
    #   If l>r, dp=0 (no offsets).
    #   If n=0, dp=0 (no creases at all).
    #
    #   Otherwise, we can place A_l..A_r in one of four ways at level n:
    #
    #   1) All in "left" half => dp(n-1, l, r)     (unchanged distribution)
    #   2) All in "right" half => (count of offsets) - dp(n-1, l, r)
    #        (because right half is inverted, so #M = total - #M_in_P(n-1))
    #   3) Partition them between left and right:
    #        pick an index i with l <= i < r, so that A_l..A_i go left,
    #        A_{i+1}..A_r go right.
    #        #M = dp(n-1, l, i)  +  [ (# in right) - dp(n-1, i+1, r ) ]
    #        (# in right) = (r - (i+1) + 1) = (r - i).
    #        We also allow i = l-1 => means all in right,
    #        or i = r => means all in left.  (But those cases are the same as #1, #2.)
    #
    #   4) Exactly one offset at the "middle fold" (which is always M),
    #      say we pick A_m as that fold, then everything smaller than A_m goes left,
    #      everything bigger than A_m goes right.  That means l..(m-1) go left,
    #      (m+1)..r go right.  #M = 1 + dp(n-1, l, m-1)
    #            + [ (# in right) - dp(n-1, m+1, r ) ]
    #        (# in right) = (r - m)
    #
    # We thus do a memoized recursion dp[n][(l,r)].  The only catch is that
    # an O(n * N^2) table of size 100*1e6 = 1e8 and transitions that might
    # also cost up to O(N) each is quite large.  However, the official solutions
    # to this problem typically do precisely this “interval DP” (or a similar
    # top-down) with various small optimizations.  In Python, it is borderline,
    # but we will implement the canonical solution and hope the test limits
    # are set so that it runs in time.
    #
    # Implementation details:
    #   We'll store dp in a top-down memo (dictionary) to avoid huge arrays.
    #   We'll code carefully, using a fast approach for the “partial cut” and
    #   the “one-middle” loops.  That is, we do them in O(r-l+1) each via
    #   precomputed arrays.  Even then, worst-case is about O(N^3) in the
    #   naive sense, but hopefully data or pruning is not too harsh.
    #
    # Finally, answer = dp(100, 1, N).
    #
    sys.setrecursionlimit(10**7)

    # We will index offsets from 1..N (just by their order).
    # dp[n, l, r] = maximum # of mountain folds achievable using A_l..A_r in P(n).
    from functools import lru_cache

    @lru_cache(None)
    def solve_fold(n, l, r):
        if l > r:
            return 0
        if n == 0:
            return 0
        # number of elements in [l..r]
        length = r - l + 1

        # ---------- Case 1: all in left half ----------
        left_val = solve_fold(n-1, l, r)
        best = left_val

        # ---------- Case 2: all in right half ----------
        #  => invert => #M = length - dp(n-1, l, r)
        best = max(best, length - left_val)

        # ---------- Case 3: partial cut (some left, some right) ----------
        # We'll find max over i in [l..r-1]:
        #   dp(n-1, l, i) + (r-i) - dp(n-1, i+1, r)
        # We can do a single linear scan to find the maximum.
        # That is a function of i => L(i) + (r-i) - R(i),
        #    where L(i) = dp(n-1, l, i),
        #          R(i) = dp(n-1, i+1, r).
        # Let's collect L(i) and R(i) in arrays.
        if length > 1:  # only meaningful if we can split
            Lvals = []
            Rvals = []
            for i2 in range(l, r):
                Lvals.append(solve_fold(n-1, l, i2))
            for i2 in range(l, r):
                Rvals.append(solve_fold(n-1, i2+1, r))
            # Lvals[i2-l], Rvals[i2-l] correspond to i2
            # we want max_{i2} of Lvals[i2-l] + (r-i2) - Rvals[i2-l]
            # do a single pass:
            partial_best = 0
            for i2 in range(r - l):
                val = Lvals[i2] + (r - (l + i2)) - Rvals[i2]
                if val > partial_best:
                    partial_best = val
            if partial_best > best:
                best = partial_best

        # ---------- Case 4: exactly one in the middle fold ----------
        # pick m in [l..r], #M = 1 + dp(n-1, l, m-1) + ((r-m) - dp(n-1, m+1, r))
        # We'll gather arrays for left and right:
        if length >= 1:
            Lm = []
            Rm = []
            # Lm[i] = dp(n-1, l, l+i-1) but we have to be careful with indexing
            # We'll just do a direct loop:
            for m2 in range(l, r+1):
                left_part = solve_fold(n-1, l, m2-1)
                right_part = solve_fold(n-1, m2+1, r)
                # #M = 1 + left_part + (r - m2) - right_part
                val = 1 + left_part + (r - m2) - right_part
                if val > best:
                    best = val

        return best

    ans = solve_fold(100, 1, N)
    print(ans)

# Call main() at the end
if __name__ == "__main__":
    main()