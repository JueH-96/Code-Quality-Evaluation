class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # We'll use a helper that, given a target maximum run length L,
        # computes the minimum number of flips required to turn s into a binary string
        # in which no contiguous block of identical characters has length > L.
        # We use dynamic programming.
        # For each position, maintain two DP arrays:
        #   dp0[r] = minimum cost (number of flips) so far if we ended with a '0' and the current consecutive run of 0's has length r.
        #   dp1[r] = similarly for ending with '1'.
        # r ranges from 1 to L (because run length 0 isn’t used).
        # When processing a new character s[i], we have two decisions from a given state:
        #    1. Continue the same run (only if r < L).
        #    2. Flip the bit (if needed) to start a new run (with run length 1) of the opposite bit.
        #
        # We'll then binary search L from 1 to n.

        import math
        INF = math.inf

        def feasible(L: int) -> bool:
            # dp0 and dp1 are arrays of length L+1 (we use indices 1...L)
            dp0 = [INF] * (L + 1)
            dp1 = [INF] * (L + 1)
            # Initialize for the first character at index 0
            dp0[1] = 0 if s[0] == '0' else 1
            dp1[1] = 0 if s[0] == '1' else 1
            
            # Process each subsequent character of s
            for i in range(1, n):
                ndp0 = [INF] * (L + 1)
                ndp1 = [INF] * (L + 1)
                # Extend from states ending with '0'
                for r in range(1, L + 1):
                    cost = dp0[r]
                    if cost == INF:
                        continue
                    # Option 1: Continue with '0' (only if r < L, otherwise would exceed the allowed run length)
                    if r < L:
                        new_r = r + 1
                        add_cost = 0 if s[i] == '0' else 1
                        new_cost = cost + add_cost
                        if new_cost < ndp0[new_r]:
                            ndp0[new_r] = new_cost
                    # Option 2: Switch bit: choose '1' at position i (this resets the run length to 1)
                    add_cost = 0 if s[i] == '1' else 1
                    new_cost = cost + add_cost
                    if new_cost < ndp1[1]:
                        ndp1[1] = new_cost
                # Extend from states ending with '1'
                for r in range(1, L + 1):
                    cost = dp1[r]
                    if cost == INF:
                        continue
                    # Option 1: Continue with '1' (only if r < L)
                    if r < L:
                        new_r = r + 1
                        add_cost = 0 if s[i] == '1' else 1
                        new_cost = cost + add_cost
                        if new_cost < ndp1[new_r]:
                            ndp1[new_r] = new_cost
                    # Option 2: Switch bit: choose '0' at position i (resets run length to 1)
                    add_cost = 0 if s[i] == '0' else 1
                    new_cost = cost + add_cost
                    if new_cost < ndp0[1]:
                        ndp0[1] = new_cost
                dp0, dp1 = ndp0, ndp1
            # The best overall cost is the minimum cost among all states
            best = min(min(dp0), min(dp1))
            return best <= numOps

        # Binary search for the minimum L (from 1 to n) that is achievable with at most numOps flips.
        lo, hi = 1, n
        ans = n
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans