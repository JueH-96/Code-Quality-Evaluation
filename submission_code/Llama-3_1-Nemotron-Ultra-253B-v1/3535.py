from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        # Initialize the DP array for the first element
        prev_dp = [1] * (nums[0] + 1)
        for i in range(1, n):
            delta = max(0, nums[i] - nums[i-1])
            current_dp = [0] * (nums[i] + 1)
            for v_prev in range(len(prev_dp)):
                if prev_dp[v_prev] == 0:
                    continue
                min_v_curr = v_prev + delta
                max_v_curr = nums[i]
                if min_v_curr > max_v_curr:
                    continue
                # Add to all valid v_curr positions
                for v_curr in range(min_v_curr, max_v_curr + 1):
                    current_dp[v_curr] = (current_dp[v_curr] + prev_dp[v_prev]) % MOD
            prev_dp = current_dp
            # If no valid ways, return 0 early
            if not any(prev_dp):
                return 0
        return sum(prev_dp) % MOD