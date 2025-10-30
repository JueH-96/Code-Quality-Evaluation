class Solution:
    def makeStringGood(self, s: str) -> int:
        # Pre‐compute frequency for each letter a..z.
        freq = [0]*26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        T = len(s)  # total count
        
        # We'll use a sufficiently high INF.
        INF = 10**9
        
        # dp_for_X(X) returns the minimum extra cost (computed by our DP)
        # if in every letter we set "used" letter final count = X.
        def dp_for_X(X: int) -> int:
            size = T + 1  # possible carry values 0 .. T
            # Base: after letter 'z' (i=26),
            # we delete any leftover carry (cost = r)
            dp = list(range(size))
            # Process letters in reverse order (from 'z' backwards to 'a')
            # (we use indices 0..25; letter 0 corresponds to 'a')
            for i in range(25, -1, -1):
                new_dp = [INF] * size
                cnt = freq[i]
                # For every possible carry r, if it is reachable,
                # update for the two options.
                for r in range(size):
                    cur = dp[r]
                    if cur == INF:
                        continue
                    a = r + cnt
                    # Option 1: Do not use letter i.
                    # We “convert” all a copies upward (cost = a) and new carry becomes a.
                    if a < size:
                        tmp = cur + a
                        if tmp < new_dp[a]:
                            new_dp[a] = tmp
                    # Option 2: Use letter i (set its final frequency = X).
                    # If a >= X, we “convert” the surplus: cost = a - X, new carry becomes a - X.
                    # Otherwise, we must insert (X - a) characters (cost = X - a) and new carry becomes 0.
                    if a >= X:
                        new_r = a - X
                        tmp = cur + (a - X)
                    else:
                        new_r = 0
                        tmp = cur + (X - a)
                    if tmp < new_dp[new_r]:
                        new_dp[new_r] = tmp
                dp = new_dp
            # Finally, we must “delete” any remaining carry.
            best = INF
            for r in range(size):
                cand = dp[r] + r
                if cand < best:
                    best = cand
            return best

        # We also consider the possibility of deleting all characters (empty string)
        ans = T

        # Now search for the best target frequency X in the range [1, T].
        # (Because if we delete all, cost = T; so the answer is at most T.)
        # We use a "ternary–like" search: because our cost F(X) = dp_for_X(X)
        # is (almost) convex so that the minimum is achieved for a relatively small range.
        lo, hi = 1, T
        best_cost = INF
        # Ternary–style search narrowing the interval until at most 10 values remain.
        while hi - lo > 10:
            mid1 = lo + (hi - lo) // 3
            mid2 = hi - (hi - lo) // 3
            f1 = dp_for_X(mid1)
            f2 = dp_for_X(mid2)
            if f1 < f2:
                hi = mid2
            else:
                lo = mid1
            if f1 < best_cost:
                best_cost = f1
            if f2 < best_cost:
                best_cost = f2
        # Finally, just check the remaining candidate X values.
        for X in range(lo, hi + 1):
            cur = dp_for_X(X)
            if cur < best_cost:
                best_cost = cur
        # answer is the minimum over our DP and the possibility of simply deleting all
        if best_cost < ans:
            ans = best_cost
        return ans