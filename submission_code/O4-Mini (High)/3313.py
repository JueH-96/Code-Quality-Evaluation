class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # prefix_sum[i] = sum of nums[0..i-1]
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]

        # We will do DP over number of chosen segments t = 0..k.
        # dp_prev[i] = max strength using t-1 segments in prefix ending at or before i
        # Initialize for t=0: zero segments => strength = 0
        dp_prev = [0] * (n + 1)

        # A large negative sentinel for impossible states
        INF = 10**30
        NEG_INF = -INF

        # For each t = 1..k, compute dp_cur from dp_prev
        for t in range(1, k + 1):
            # coefficient for the t-th chosen subarray
            # sign = 1 if t is odd, -1 if t is even
            sign = 1 if (t % 2) == 1 else -1
            coeff = sign * (k - t + 1)

            dp_cur = [NEG_INF] * (n + 1)
            # We cannot pick t segments from an empty prefix
            dp_cur[0] = NEG_INF

            # We'll maintain maxM = max over j< i of (dp_prev[j] - coeff*prefix[j])
            maxM = NEG_INF

            for i in range(1, n + 1):
                # Update maxM with j = i-1
                val = dp_prev[i - 1] - coeff * prefix[i - 1]
                if val > maxM:
                    maxM = val
                # If we end the t-th segment at i, its contribution is coeff*sum(segment)
                # sum(segment) = prefix[i] - prefix[j], and we add dp_prev[j]
                # So candidate = coeff*prefix[i] + maxM
                cand = coeff * prefix[i] + maxM
                # Either we end our t-th segment at i, or we don't use i (carry over dp_cur[i-1])
                dp_cur[i] = dp_cur[i - 1] if dp_cur[i - 1] > cand else cand

            # Move to next t
            dp_prev = dp_cur

        # The answer is dp[k][n], i.e. dp_prev[n] after t=k
        return dp_prev[n]