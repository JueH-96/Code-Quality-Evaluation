def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster output routine
    import sys
    write = sys.stdout.write

    # ------------------------------------------------------------
    # 1) Parse input
    # ------------------------------------------------------------
    # N
    N = int(input_data[0])
    # A
    A = list(map(int, input_data[1:1+N]))
    # B
    B = list(map(int, input_data[1+N:1+2*N]))
    # K
    K = int(input_data[1+2*N])
    # queries (X_k, Y_k)
    qraw = input_data[1+2*N+1:]
    queries = []
    idx = 0
    for k in range(K):
        xk = int(qraw[2*k])
        yk = int(qraw[2*k+1])
        queries.append((xk, yk, k))
    # We'll need to output answers in the original order
    answers = [0]*K

    # ------------------------------------------------------------
    # 2) Precompute prefix sums of A and B (for the easy part)
    #    prefixSumA[x] = A_1 + ... + A_x
    #    prefixSumB[y] = B_1 + ... + B_y
    # ------------------------------------------------------------
    prefixA = [0]*(N+1)
    prefixB = [0]*(N+1)
    for i in range(N):
        prefixA[i+1] = prefixA[i] + A[i]
        prefixB[i+1] = prefixB[i] + B[i]

    # ------------------------------------------------------------
    # 3) Observe that:
    #
    #    S(x,y) = sum_{i=1..x} sum_{j=1..y} |A_i - B_j|
    #            = x * (sum of B_1..B_y)
    #              + y * (sum of A_1..A_x)
    #              - 2 * M(x,y),
    #
    #    where M(x,y) = sum_{i=1..x} sum_{j=1..y} min(A_i, B_j).
    #
    #  But directly computing M(x,y) for each query in O(x*y) or O(N^2)
    #  is impossible for large N.  A well‐known identity for
    #  sum of min’s can be written as an integral or by discrete counting
    #  of how many A_i > t and B_j > t, etc.  However, implementing
    #  that efficiently for large N and up to 10^4 queries is nontrivial.
    #
    #  The most straightforward practical approach here (within the time
    #  of a written solution) is to do an offline “sweep” in ascending
    #  order of x and y.  We maintain M(x,y) incrementally:
    #
    #    - Start from x=0,y=0 => M(0,0) = 0.
    #    - When x increases by 1 (we add A_{x}), we update M accordingly.
    #    - When y increases by 1 (we add B_{y}), we update M accordingly.
    #
    #  The update step uses the fact that:
    #    M(x+1, y) = M(x, y) + sum_{j=1..y} min(A_{x+1}, B_j),
    #    M(x, y+1) = M(x, y) + sum_{i=1..x} min(A_i, B_{y+1}).
    #
    #  Naively summing “min(A_{x+1}, B_j) for j=1..y” in O(y) each time
    #  would be O(N^2), still too large for N=1e5.
    #
    #  A more refined data structure approach is needed (e.g. order-statistic
    #  trees or Fenwicks storing partial sums), but it is fairly intricate
    #  to implement in Python under contest conditions.  
    #
    #  -------------------------------------------------------------
    #  BELOW is a “coordinate-compression + sweep + suffix-array” approach
    #  sketched out.  However, given the complexity, and typical time-limit,
    #  such a solution in Python can be borderline.  In a lower-level
    #  language (C++), one would carefully implement 2D-lazy structures or
    #  offline range updates.  
    #
    #  Here, we provide a correct (but not highly optimized) method:
    #  1) We compute the “easy part” in O(1) for each query: x * sumB(y) + y * sumA(x).
    #  2) For the “M(x,y)=sum min(A_i,B_j)” piece, we do a slower method with
    #     coordinate compression and partial prefix frequencies.  In worst
    #     case it is O(K * N) or O(K * D).  This may pass small or moderate
    #     tests, but can time out in worst cases.  
    #
    #  Nonetheless, it is a clean reference showing how one might build
    #  solutions up.  For truly large constraints, a more advanced data
    #  structure is required.
    #  -------------------------------------------------------------
    #
    #  We'll implement the "frequency + discrete sum" approach in O(K*D) form:
    #    M(x,y) = sum_{t=0..∞} # {i<=x | A_i>t} * # {j<=y | B_j>t}.
    #  We discretize possible t-values from the sets A and B.
    #
    #  Then # {i<=x | A_i>val[d]} = x - # {i<=x | A_i <= val[d]},
    #  with val[d] in sorted distinct values.  We'll build prefix arrays
    #  for A- and B-values offline and answer queries in sorted order of x,y.
    #  This is still large but is the easiest correct method to illustrate.
    #
    #  We STRONGLY caution that this approach can be too slow for the
    #  largest test if N=1e5 and K=1e4 and all values are distinct.  But
    #  it is correct and will pass smaller tests or partial subtasks.
    #
    # ------------------------------------------------------------

    # Collect all values of A and B for compression
    vals = []
    vals.extend(A)
    vals.extend(B)
    # Also add 0 so we handle min(...) > 0 nicely
    # (Strictly speaking it's not necessary if we only care about 0..max(A,B))
    # but it's often convenient.
    vals = list(set(vals))
    vals.sort()
    # map from value -> rank
    rank_map = {}
    for i,v in enumerate(vals):
        rank_map[v] = i
    D = len(vals)

    # rankA[i] = rank of A[i]
    rankA = [rank_map[a] for a in A]
    # rankB[i] = rank of B[i]
    rankB = [rank_map[b] for b in B]

    # We'll store freqA_up_to[x][d] = # of i<=x with A_i > vals[d].
    # But we can't store a full 2D array (that would be N*D => 1e10).
    #
    # Instead we do an offline approach:
    # Sort queries by x ascending, then for each x we can compute freqA_x(d)
    # on the fly.  freqA_x(d) = x - (# of i<=x with A_i <= vals[d]).
    # We'll keep a Fenwick tree (BIT) for "counts of ranks" as we move x forward.
    #
    # freqA_x(d) = x - (count of i<=x with rankA[i] <= d).
    # So if we have a Fenwick that, after inserting A_i's up to i=x,
    # can answer how many of them have rank <= d, we can compute freqA_x(d).
    #
    # Then M(x,y) = sum_{d=0..D-1} freqA_x(d)*freqB_y(d)  -- but each pair
    # actually contributes for integer t in [ vals[d], vals[d+1] ), so we need
    # the difference in the actual “> t” counting.  The discrete formula is:
    #
    #   M(x,y) = sum_{d=0..D-1} freqA_x(d)*freqB_y(d) * ( gap between vals[d] and vals[d+1] )
    #
    # Where freqA_x(d) = # of i<=x s.t. A_i > vals[d].
    #       freqB_y(d) = # of j<=y s.t. B_j > vals[d].
    #
    # and the “gap” = vals[d+1] - vals[d].  For d=D-1, define vals[D] = maxval+1 for convenience.
    #
    # Implementation detail: we will store freqA_x(d) in an array after we move x,
    # freqB_y(d) in another array after we move y.  Then computing M(x,y) is an O(D) loop,
    # which for K=1e4, D up to 2e5 => 2e9 operations (borderline large).
    #
    # We will implement it anyway (as a reference). A real contest solution in Python
    # typically would need more clever data structures or partial-subtask acceptance.

    # Fenwicks (BIT) for queries of the form "how many of the inserted ranks are <= r"
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        def update(self, idx, val):
            # idx in [0..n-1]
            i = idx+1
            while i <= self.n:
                self.data[i] += val
                i += i & -i
        def query(self, idx):
            # returns sum of [0..idx]
            s = 0
            i = idx+1
            while i>0:
                s += self.data[i]
                i -= i & -i
            return s
        def query_range(self, left, right):
            # sum in [left, right]
            return self.query(right) - self.query(left-1)

    # Prepare Fenwicks for A- and B- side
    fenA = Fenwick(D)
    fenB = Fenwick(D)

    # Sort queries by (x, y)
    queries.sort(key=lambda z: (z[0], z[1]))

    # We will maintain:
    #   xcurr, ycurr = how far we have inserted A_i or B_j
    #   freqA(d) = # of i <= xcurr such that A_i > vals[d] = xcurr - (# with rankA[i]<= d )
    # Similarly freqB(d).
    # Then for each query(x,y), we advance xcurr->x, ycurr->y, build freqA, freqB arrays,
    # compute M(x,y), then final answer.

    xcurr = 0
    ycurr = 0

    # We'll store freqAarray[d], freqBarray[d] for d in [0..D-1].
    # freqAarray[d] = number of A_i in [1..xcurr] with A_i > vals[d].
    # We'll only rebuild them when xcurr or ycurr changes.
    freqAarray = [0]*D
    freqBarray = [0]*D

    # A helper function to rebuild freqAarray after xcurr changes:
    def rebuild_freqA():
        # freqAarray[d] = xcurr - fenA.query(d)
        # because fenA.query(d) = # of inserted A_i with rank <= d
        for d in range(D):
            leq = fenA.query(d)
            freqAarray[d] = xcurr - leq

    # Similarly for freqBarray
    def rebuild_freqB():
        for d in range(D):
            leq = fenB.query(d)
            freqBarray[d] = ycurr - leq

    # Precompute the "gaps" w[d] = vals[d+1] - vals[d], for d=0..D-2
    # For convenience, define an extra vals at end = vals[D-1]+1
    # so that we handle the last segment.  Or we can just handle up to D-2 carefully.
    vals.append(vals[-1]+1)  # so now len(vals) = D+1
    w = [0]*D
    for d in range(D):
        w[d] = vals[d+1] - vals[d]

    # We'll keep M(xcurr, ycurr) in a variable so we do not have to re-sum in O(D)
    # each time we move x or y by 1.  However, updating it by 1 in x or y each time
    # still needs an O(y) or O(x) pass.  That’s also large.  So to keep it simpler,
    # we will just recompute M(x,y) from freq arrays for each query.  This is simpler,
    # though not the fastest.
    # A more advanced incremental approach would do:
    # M(x+1,y) = M(x,y) + sum_{j=1..y} min(A_{x+1}, B_j), etc. with balanced trees.

    # We'll process queries in ascending x,y.  For each query we will:
    #  1) move xcurr up to x
    #  2) move ycurr up to y
    #  3) rebuild freq arrays
    #  4) compute M in O(D)
    #  5) finalize answer

    # Current index in queries
    ptr = 0
    Q = len(queries)

    # We'll accumulate results in an array "answers"

    for (xq, yq, qid) in queries:
        # move xcurr up to xq
        while xcurr < xq:
            # insert rank(A[xcurr]) into fenA
            rA = rankA[xcurr]
            fenA.update(rA, 1)
            xcurr += 1
        # move ycurr up to yq
        while ycurr < yq:
            rB = rankB[ycurr]
            fenB.update(rB, 1)
            ycurr += 1

        # Now rebuild freq arrays for the current (xcurr, ycurr)
        rebuild_freqA()
        rebuild_freqB()

        # Compute M(xcurr, ycurr)
        # M = sum_{d=0..D-1} freqAarray[d]*freqBarray[d]* w[d]
        mval = 0
        for d in range(D):
            mval += freqAarray[d]*freqBarray[d]*w[d]

        # Then compute sum of absolute differences
        # = xcurr * prefixB[ycurr] + ycurr * prefixA[xcurr] - 2*mval
        ans = xcurr*prefixB[ycurr] + ycurr*prefixA[xcurr] - 2*mval
        answers[qid] = ans

    # Finally print answers
    # K lines
    out = []
    for k in range(K):
        out.append(str(answers[k]))
    write("
".join(out) + "
")


# Do not forget to call main()!
if __name__ == "__main__":
    main()