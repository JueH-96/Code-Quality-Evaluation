class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # Quick base
        if n == 0:
            return 0
        # Check feasibility for a given max-run length L
        def can(L):
            # INF caps any cost > numOps
            INF = numOps + 1
            # dp_prev0[r] = min flips to build prefix ending with r repeats of '0'
            # dp_prev1[r] = min flips to build prefix ending with r repeats of '1'
            dp_prev0 = [INF] * (L + 1)
            dp_prev1 = [INF] * (L + 1)
            # Initialize for i = 0
            # if we choose t[0] = '0'
            cost0 = 0 if s[0] == '0' else 1
            if cost0 <= numOps:
                dp_prev0[1] = cost0
            # if we choose t[0] = '1'
            cost1 = 0 if s[0] == '1' else 1
            if cost1 <= numOps:
                dp_prev1[1] = cost1
            # If the string has length 1, just check the initialization
            if n == 1:
                return (dp_prev0[1] <= numOps) or (dp_prev1[1] <= numOps)
            # Build DP for i = 1 .. n-1
            for i in range(1, n):
                dp_cur0 = [INF] * (L + 1)
                dp_cur1 = [INF] * (L + 1)
                # Track minimum newly filled cost in dp_cur
                newMin0 = INF
                newMin1 = INF
                # cost to place '0' or '1' at position i
                flip0 = 1 if s[i] == '1' else 0
                flip1 = 1 if s[i] == '0' else 0
                # Extend runs of '0' and compute min of dp_prev0
                minPrev0 = dp_prev0[L]
                for run in range(1, L):
                    c0 = dp_prev0[run]
                    # update global min for transitions
                    if c0 < minPrev0:
                        minPrev0 = c0
                    # extend the '0'-run
                    nc = c0 + flip0
                    if nc < dp_cur0[run+1]:
                        dp_cur0[run+1] = nc
                        if nc < newMin0:
                            newMin0 = nc
                # Extend runs of '1' and compute min of dp_prev1
                minPrev1 = dp_prev1[L]
                for run in range(1, L):
                    c1 = dp_prev1[run]
                    if c1 < minPrev1:
                        minPrev1 = c1
                    nc = c1 + flip1
                    if nc < dp_cur1[run+1]:
                        dp_cur1[run+1] = nc
                        if nc < newMin1:
                            newMin1 = nc
                # Start a new '1'-run at i (coming from a '0'-run)
                nc = minPrev0 + flip1
                if nc < dp_cur1[1]:
                    dp_cur1[1] = nc
                    if nc < newMin1:
                        newMin1 = nc
                # Start a new '0'-run at i (coming from a '1'-run)
                nc = minPrev1 + flip0
                if nc < dp_cur0[1]:
                    dp_cur0[1] = nc
                    if nc < newMin0:
                        newMin0 = nc
                # If no state is within budget, fail early
                if newMin0 > numOps and newMin1 > numOps:
                    return False
                dp_prev0, dp_prev1 = dp_cur0, dp_cur1
            # After finishing, check if any end-state is within budget
            # We can use the last newMin0/1 or scan dp_prev arrays:
            # Here scanning is simple:
            if min(dp_prev0[1:]) <= numOps or min(dp_prev1[1:]) <= numOps:
                return True
            return False

        # Binary search on the answer L in [1..n]
        lo, hi = 1, n
        ans = n
        while lo <= hi:
            mid = (lo + hi) // 2
            if can(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans