from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize DP for i=0
        prev_dp = [1] * (nums[0] + 1)
        # Compute prefix sums for i=0
        prev_prefix = [0] * (nums[0] + 2)
        for k in range(nums[0] + 1):
            prev_prefix[k+1] = (prev_prefix[k] + prev_dp[k]) % MOD
        
        # Iterate through i from 1 to n-1
        for i in range(1, n):
            current_num = nums[i]
            prev_num = nums[i-1]
            s = max(0, current_num - prev_num)
            upper_k = current_num
            curr_dp = [0] * (upper_k + 1)
            # Compute DP[i][k] for each k from 0 to nums[i]
            for k in range(upper_k + 1):
                if k >= s:
                    upper_prev_k = min(k - s, prev_num)
                    curr_dp[k] = prev_prefix[upper_prev_k + 1]
                else:
                    curr_dp[k] = 0
            # Update prev_dp and compute new prefix sums
            prev_dp = curr_dp
            # Compute prefix sums for the current DP
            prev_prefix = [0] * (upper_k + 2)
            for k in range(upper_k + 1):
                prev_prefix[k+1] = (prev_prefix[k] + prev_dp[k]) % MOD
        
        # The answer is the sum of prev_dp[k] for k in 0 to nums[n-1], modulo 10^9+7
        answer = sum(prev_dp) % MOD
        return answer