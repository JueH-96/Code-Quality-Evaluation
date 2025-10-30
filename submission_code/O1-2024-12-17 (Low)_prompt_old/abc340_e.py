def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO readers
    # Parsing inputs
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))
    
    # ----------------------------------------------------------------
    # We need to simulate the following process M times in order:
    #  1) Let count = (current number of balls in box B_i).
    #  2) Set that box's number of balls to 0.
    #  3) Distribute 'count' balls, one each to consecutive boxes
    #     (B_i+1) mod N, (B_i+2) mod N, ..., (B_i+count) mod N.
    #
    # But since N can be up to 2e5 and each A_i can be up to 1e9,
    # a naive approach is impossible (it could take ~1e14 operations).
    #
    # Instead, we observe that putting 'count' balls one-by-one to the next
    # 'count' positions in a circular manner is equivalent to:
    #
    #   (1) Each box gets (count // N) increments (if count >= N),
    #       because we effectively wrap around multiple times.
    #   (2) Plus one more increment for the first (count % N) boxes
    #       in the circular order, starting from (B_i+1) mod N.
    #
    # We can track the "global" increments (the (count // N) part) via
    # a single integer "totalAdd" which counts how many times every box
    # has been incremented so far.
    #
    # Then for the "partial" increments (the (count % N) leftover),
    # we can use a data structure that supports:
    #   - Range updates (on a circular array)
    #   - Point queries (to find how many increments a single box has)
    #
    # The reason we need point queries is that on each of the M steps,
    # we need to find how many balls are in box B_i "now", i.e.:
    #     X[B_i] = A[B_i] + totalAdd + partial_increments_so_far[B_i].
    #
    # After we determine count = X[B_i], we then set X[B_i] to 0,
    # meaning we must also "remove" those count balls from B_i so that
    # future queries see that B_i has 0 new increments. We can accomplish
    # that by applying a "range update" of length 1 (i.e., a point update)
    # subtracting 'count' from B_i in our data structure. 
    #
    # Then we do the partial increment of (count % N) boxes starting
    # from (B_i + 1). 
    #
    # A Fenwick Tree (Binary Indexed Tree) can manage:
    #   - range updates
    #   - point queries
    # by using the standard "Fenwicks-of-Fenwicks" or "two-Fenwicks" technique.
    #
    # The final formula for the value at index i after all partial updates is:
    #   X[i] = A[i] + totalAdd + (fenw_value(i))
    #
    # Where fenw_value(i) is the net contribution of all partial range updates
    # to index i. We also must make sure to handle circular ranges properly.
    #
    # Complexity: O((N + M) log N) which is feasible for N, M <= 2e5.
    #
    # Implementation details:
    #
    #   We define two Fenwicks, fenw1 and fenw2, each of size N+1 (for 0-based indexing):
    #     - To do a range update [L..R] (inclusive) of +v (0 <= L <= R < N):
    #         addFenw(fenw1, L, +v)
    #         addFenw(fenw1, R+1, -v)   if (R+1 <= N-1)
    #         addFenw(fenw2, L, +v * L)
    #         addFenw(fenw2, R+1, -v * (R+1))  if (R+1 <= N-1)
    #
    #     - The value at index i (0 <= i < N) after these updates is:
    #         get(i) = i * sumFenw(fenw1, i) - sumFenw(fenw2, i)
    #       where sumFenw(...) gives prefix sums up to i.
    #
    #   For a circular update [start..end] mod N with +v, where length is <= N:
    #     if start <= end:
    #         rangeAdd(start, end, +v)
    #     else:
    #         rangeAdd(start, N-1, +v)
    #         rangeAdd(0, end, +v)
    #
    # We must be careful about boundary checks.
    #
    # Steps in solve():
    #   1) Initialize fenw1, fenw2 with all zeros.
    #   2) totalAdd = 0  (accumulate full wraps)
    #   3) For i in [0..M-1]:
    #         # get current X[B_i]
    #         count = A[B_i] + totalAdd + getFenwValue(B_i)
    #
    #         # remove 'count' from box B_i
    #         rangeAdd(B_i, B_i, -count)   # This sets box B_i's fenw contribution to 0 extra.
    #         A[B_i] = 0                  # so that direct A[B_i] won't repeat
    #
    #         # distribute 'count' balls
    #         q, r = divmod(count, N)
    #         totalAdd += q
    #
    #         if r > 0:
    #             start = (B_i + 1) % N
    #             end = (B_i + r) % N
    #             if start <= end:
    #                 rangeAdd(start, end, 1)
    #             else:
    #                 # wraps around
    #                 rangeAdd(start, N-1, 1)
    #                 rangeAdd(0, end, 1)
    #
    #   4) After all M steps, compute final X[i] for i in [0..N-1]:
    #         X[i] = A[i] + totalAdd + getFenwValue(i)
    #      Print them.
    #
    # Let's implement it.
    # ----------------------------------------------------------------

    sys.setrecursionlimit(10**7)

    # Fenwicks (BIT) for 0-based index up to N (we'll use size N+1 internally).
    # We'll store them in lists of length N+1; index 0 is unused or used carefully.

    fenw1 = [0]*(N+1)
    fenw2 = [0]*(N+1)

    # Fenwicks helper functions (all indices 0-based for the array positions).
    # We'll define "addFenw(bit, pos, val)" that adds 'val' at index pos (0-based).
    # Implementation detail: internally we do pos+1 for the Fenw, and while i<=N do ...
    def fenw_add(bit, idx, val):
        """ Add 'val' to fenw[] at index idx (0-based for the problem),
            but we store in fenw using 1-based indexing internally.
        """
        i = idx + 1
        while i <= N:
            bit[i] += val
            i += i & -i

    def fenw_sum(bit, idx):
        """ Returns Fenw prefix sum from 0..idx inclusive (0-based).
            Internally uses 1-based indexing.
        """
        res = 0
        i = idx + 1
        while i > 0:
            res += bit[i]
            i -= i & -i
        return res

    def fenw_range_add(L, R, val):
        """ Range update [L..R] with +val, in the Fenwicks. """
        # For fenw1:
        fenw_add(fenw1, L, val)
        if R+1 < N:
            fenw_add(fenw1, R+1, -val)
        # For fenw2:
        fenw_add(fenw2, L, val*L)
        if R+1 < N:
            fenw_add(fenw2, R+1, -val*(R+1))

    def getFenwValue(idx):
        """ Get the accrued value at index 'idx' from fenw1/fenw2. """
        # i * sumFenw(fenw1, i) - sumFenw(fenw2, i)
        s1 = fenw_sum(fenw1, idx)
        s2 = fenw_sum(fenw2, idx)
        return idx*s1 - s2

    # Now let's implement the main simulation.
    totalAdd = 0

    for i in range(M):
        b = B[i]
        # current number of balls in box b
        count = A[b] + totalAdd + getFenwValue(b)
        # remove 'count' from box b
        if count != 0:
            # rangeAdd(b, b, -count)
            fenw_range_add(b, b, -count)
            A[b] = 0

            # distribute
            q, r = divmod(count, N)
            totalAdd += q
            if r > 0:
                start = (b + 1) % N
                end   = (b + r) % N
                if start <= end:
                    fenw_range_add(start, end, 1)
                else:
                    # we wrap the end
                    fenw_range_add(start, N-1, 1)
                    fenw_range_add(0, end, 1)

    # Finally compute the result for each box
    # X[i] = A[i] + totalAdd + getFenwValue(i)
    out = []
    for i in range(N):
        val = A[i] + totalAdd + getFenwValue(i)
        out.append(str(val))

    print(" ".join(out))

# Let's call solve() (as required).
if __name__ == "__main__":
    solve()