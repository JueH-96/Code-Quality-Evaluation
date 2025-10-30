class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        import sys
        import bisect

        # ---------------------------------------------------------
        # 1) Sort the input intervals by their start coordinate.
        #    Each interval is (l, r, c).
        # ---------------------------------------------------------
        intervals = sorted(coins, key=lambda x: x[0])
        n = len(intervals)

        # ---------------------------------------------------------
        # 2) Collect all "boundary" points in a set E:
        #    For each interval [l, r], we will add:
        #      l      (start of interval)
        #      r + 1  (one past the end, for half-open segment logic)
        #
        #    We will also ensure we have at least one boundary so
        #    we can safely build partial sums.  (Though n>=1 per constraints.)
        # ---------------------------------------------------------
        E = set()
        for l, r, _ in intervals:
            E.add(l)
            E.add(r + 1)

        # Convert to a sorted list
        Elist = sorted(E)  # This will have length m <= 2*n

        # ---------------------------------------------------------
        # 3) Build a piecewise-constant description of the coin
        #    distribution using these boundaries.  For j in [0..m-2],
        #    the segment covers [Elist[j], Elist[j+1]) at a constant
        #    coin value val[j].
        #
        #    Because intervals are non-overlapping, each point on
        #    the line can be covered by at most one interval.
        # ---------------------------------------------------------
        m = len(Elist)
        val = [0]*(m-1)  # val[j] will be the coin value in [Elist[j], Elist[j+1]).
        P = [0]*m        # prefix sum array of coin distribution

        i = 0  # pointer to intervals
        for j in range(m-1):
            start = Elist[j]
            end   = Elist[j+1]

            # Move 'i' forward if this segment is entirely beyond intervals[i].
            # While intervals[i].r < start, we skip that interval.
            while i < n and intervals[i][1] < start:
                i += 1

            # Now check if the current segment is inside intervals[i].
            if i < n:
                l_i, r_i, c_i = intervals[i]
                if l_i <= start <= r_i:
                    val[j] = c_i
                else:
                    val[j] = 0
            else:
                val[j] = 0

            length_j = end - start  # length of this segment
            P[j+1] = P[j] + val[j]*length_j

        # ---------------------------------------------------------
        # partial_sum(u):  returns the total coins in [ (smallest Elist[0]), u ),
        # clamped so that below Elist[0] is 0 and above Elist[-1] is the full sum.
        # ---------------------------------------------------------
        def partial_sum(u: int) -> int:
            # If u <= Elist[0], no coverage
            if u <= Elist[0]:
                return 0
            # If u >= Elist[-1], full coverage up to the last boundary
            if u >= Elist[-1]:
                return P[-1]
            # Otherwise, find j such that Elist[j] <= u < Elist[j+1]
            j = bisect.bisect_right(Elist, u) - 1
            # Add partial coverage in the current segment
            return P[j] + val[j]*(u - Elist[j])

        # ---------------------------------------------------------
        # 4) We only need to check x at "critical" points where
        #    the coverage [x, x+k) might change its overlap with intervals.
        #
        #    Those critical points are each boundary e in Elist,
        #    and also (e - k) for each e in Elist, because shifting
        #    the window so its right boundary hits an interval boundary
        #    can yield a new overlap.
        #
        #    We'll consider all such x in the range [Elist[0] - k, Elist[-1]].
        # ---------------------------------------------------------
        minX = Elist[0] - k  # below this, the window won't gain new coins
        maxX = Elist[-1]     # above this, the window starts beyond last boundary

        # Build the set of candidate x values
        X = set()
        for e in Elist:
            X.add(e)
            X.add(e - k)

        # We will keep only x in [minX, maxX]
        candidates = []
        for x in X:
            if x >= minX and x <= maxX:
                candidates.append(x)

        candidates.sort()

        # ---------------------------------------------------------
        # 5) Evaluate the sum of coins for each candidate x:
        #       sum_in_window = partial_sum(x + k) - partial_sum(x)
        #    Then take the maximum.
        # ---------------------------------------------------------
        ans = 0
        for x in candidates:
            left = x
            right = x + k
            # sum of coins in [left, right)
            curr = partial_sum(right) - partial_sum(left)
            if curr > ans:
                ans = curr

        return ans