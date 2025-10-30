from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        # Sort requirements by end_i
        req = {}
        for end, cnt in requirements:
            req[end] = cnt
        # Initialize DP
        max_inv = 400
        dp_prev = [0] * (max_inv + 1)
        dp_prev[0] = 1
        for i in range(1, n + 1):
            dp_curr = [0] * (max_inv + 1)
            # Compute prefix sums for dp_prev
            prefix = [0] * (max_inv + 2)
            for k in range(max_inv +1):
                prefix[k+1] = (prefix[k] + dp_prev[k]) % MOD
            for k in range(max_inv +1):
                j_max = min(i-1, k)
                dp_curr[k] = prefix[k +1] - prefix[k - j_max] if k - j_max >=0 else prefix[k +1]
                if dp_curr[k] < 0:
                    dp_curr[k] += MOD
                else:
                    dp_curr[k] %= MOD
            # If there's a requirement at i-1, enforce it
            end_idx = i-1
            if end_idx in req:
                target = req[end_idx]
                dp_filtered = [0] * (max_inv +1)
                if 0 <= target <= max_inv:
                    dp_filtered[target] = dp_curr[target]
                dp_curr = dp_filtered
            dp_prev = dp_curr
        # The final requirement should have end_i == n-1
        final_cnt = req.get(n-1, None)
        if final_cnt is not None and 0 <= final_cnt <= max_inv:
            return dp_prev[final_cnt]
        else:
            return 0