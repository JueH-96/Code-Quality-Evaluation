class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        We want an array nums of length n, all positive, strictly increasing,
        whose bitwise-AND is x. We must return the smallest possible value of
        nums[n-1].

        Key observations:
        1) For the AND of all elements to be exactly x, every element must have
           all the bits set that x has.  (Equivalently, for each bit i where x
           has bit i=1, that bit i must also be 1 in every element.)
        2) If at least one element has a "0" in a bit where x has "0," that keeps
           that bit out of the final AND (so it remains zero).  Picking the first
           element = x suffices to ensure that all bits not in x will appear as 0
           in at least one element.  This guarantees the overall AND remains x.
        3) Therefore, the sequence simply needs to be n distinct integers in
           strictly increasing order, each of which "covers" x in a bitwise sense:
               for every element y,  (y & x) == x.
           In other words, each element y has at least the bits of x.
        4) Among all integers ≥ 1, those that satisfy (y & x) == x form an
           increasing set.  We want the n-th smallest such integer (1-based),
           and we specifically start counting from x itself as the first such
           integer (since x & x = x).
        5) So the problem reduces to:  "Find the n-th smallest y such that
           (y & x) = x."  Equivalently, "Find the minimal y with rank(y) = n,"
           where rank(y) = number of integers ≤ y having (z & x) = x.
        6) We can do a binary search on y.  For each candidate mid, we compute
           how many z in [0..mid] satisfy (z & x) = x.  If that count >= n, we go
           left; otherwise we go right.  In the end we get the exact y where the
           count of valid z is n.
        7) The count of valid z in [0..mid] with (z & x) = x can be computed
           with a standard bitwise-DP ("digit DP") approach in O(32 * 2) per call,
           which is fast enough when combined with a ~30-step binary search
           (since n,x up to 1e8 means we might need up to ~40 bits to be safe).

        Steps to solve:
          - If n == 1, answer is just x.
          - Otherwise do a binary search for y in [x..some_upper_bound], where
            some_upper_bound can be x + (1 << 40) safely (or even (1 << 35)) 
            so we don't miss the n-th valid number.
          - In each step, compute countValid(mid, x), which returns how many 
            integers z <= mid satisfy (z & x) == x.
          - Compare that count with n to decide which half to search in.
          - Return the final low.

        Let's implement it.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # If we only need 1 element, that element must be x.
        if n == 1:
            return x

        # Digit-DP to count how many z in [0..mid] satisfy (z & x) == x.
        #
        # We'll do a standard top-down bitwise DP:
        #   - We examine bits from the most significant down to 0 (say 31..0).
        #   - "smaller" indicates whether we've already chosen a bit pattern for z
        #     that is strictly less than the corresponding prefix of mid,
        #     allowing any bits to be chosen freely afterward.
        #   - If x has bit i = 1, we must choose z_i = 1.
        #   - If x has bit i = 0, we can choose z_i = 0 or 1, subject to z <= mid.
        #   - We memoize dp[pos][smaller].

        def countValid(mid: int, x: int) -> int:
            """Count how many z in [0..mid] have (z & x) == x."""
            if mid < x:
                return 0

            # Build bit arrays for mid and x (up to 31 bits for 10^8, but let's do 32).
            bitsMid = [(mid >> i) & 1 for i in range(32)]
            bitsX   = [(x   >> i) & 1 for i in range(32)]

            # dp[pos][smaller] = number of ways to choose bits for z from pos down to 0
            # such that z <= mid (if not smaller yet), and (z & x) = x.
            # We'll use -1 to indicate "not computed."
            dp = [[-1]*2 for _ in range(32)]

            def rec(pos: int, smaller: int) -> int:
                if pos < 0:
                    # If we assigned all bits consistently, that's exactly 1 valid pattern.
                    return 1
                if dp[pos][smaller] != -1:
                    return dp[pos][smaller]

                ways = 0
                m = bitsMid[pos]   # the bit of mid at 'pos'
                bx = bitsX[pos]    # the bit of x at 'pos'

                # We want to choose bit c for z at position pos.
                # If x has bit pos=1, we must choose c=1.
                # If x has bit pos=0, c can be 0 or 1, provided we don't exceed mid.

                for c in (0, 1):
                    # If x bit is 1, c MUST be 1, so skip c=0 in that scenario.
                    if bx == 1 and c == 0:
                        continue

                    # If we are not already smaller than mid, we cannot choose c>m.
                    if smaller == 0:
                        if c > m:
                            continue
                        # If c < m, then we become strictly smaller from here on.
                        nextSmaller = 1 if c < m else 0
                    else:
                        # Already smaller, we can choose any c (0 or 1) freely.
                        nextSmaller = 1

                    ways += rec(pos - 1, nextSmaller)

                dp[pos][smaller] = ways
                return ways

            return rec(31, 0)

        # We'll do a binary search for the smallest Y >= x such that countValid(Y, x) = n.
        # Because countValid(Y, x) is non-decreasing in Y, we can do standard binary search.

        def rank_of(y: int, x: int) -> int:
            """Return how many values z <= y satisfy (z & x) == x."""
            return countValid(y, x)

        # We know n >= 2 here.
        low  = x
        high = x + (1 << 40)  # safely large upper bound

        while low < high:
            mid = (low + high) // 2
            if rank_of(mid, x) >= n:
                high = mid
            else:
                low = mid + 1

        # low is now the minimal number whose rank is n, i.e. the n-th number covering x.
        return low