class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # If we can flip every bit, we can always achieve max run = 1
        if numOps >= n:
            return 1

        INF = 10**9

        # Check if it's possible to achieve max run length <= L with <= numOps flips
        def can(L: int) -> bool:
            # dp[c][l] = min flips used to reach position i with current run char = c and run length = l
            # We only need the previous row
            dp_prev = [[INF] * (L + 2) for _ in range(2)]
            # Initialize for position 0
            # If we choose T[0] = 0
            cost0 = 1 if s[0] == '1' else 0
            dp_prev[0][1] = cost0
            # If we choose T[0] = 1
            cost1 = 1 if s[0] == '0' else 0
            dp_prev[1][1] = cost1

            for i in range(1, n):
                curch = int(s[i])
                dp_cur = [[INF] * (L + 2) for _ in range(2)]
                # best cost among all run lengths for each char
                best0 = min(dp_prev[0][1:L+1])
                best1 = min(dp_prev[1][1:L+1])
                # Try setting T[i] = 0 or 1
                for t in (0, 1):
                    flip_cost = 1 if curch != t else 0
                    # Starting a new run of length 1: take best from other char
                    if t == 0:
                        prev_best = best1
                    else:
                        prev_best = best0
                    if prev_best + flip_cost < dp_cur[t][1]:
                        dp_cur[t][1] = prev_best + flip_cost

                    # Extending a run of the same char
                    for l in range(1, L):
                        prev_cost = dp_prev[t][l]
                        if prev_cost == INF:
                            continue
                        new_cost = prev_cost + flip_cost
                        if new_cost < dp_cur[t][l + 1]:
                            dp_cur[t][l + 1] = new_cost

                dp_prev = dp_cur

            # After processing all positions, check if any state uses <= numOps flips
            best_end = min(min(dp_prev[0][1:L+1]), min(dp_prev[1][1:L+1]))
            return best_end <= numOps

        # Binary search the minimal L in [1..n]
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if can(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo