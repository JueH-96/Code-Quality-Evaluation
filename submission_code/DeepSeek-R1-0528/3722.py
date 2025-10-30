NEG_INF = -10**18

class Solution:
    def maxSum(self, nums: list, k: int, m: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        dp_prev = [0] * (n + 1)
        
        for j in range(1, k + 1):
            new_dp = [NEG_INF] * (n + 1)
            best = NEG_INF
            for i in range(n + 1):
                if i >= m:
                    s = i - m
                    candidate_val = dp_prev[s] - prefix[s]
                    if candidate_val > best:
                        best = candidate_val
                if i > 0:
                    new_dp[i] = new_dp[i - 1]
                if best != NEG_INF:
                    candidate_sum = best + prefix[i]
                    if candidate_sum > new_dp[i]:
                        new_dp[i] = candidate_sum
            dp_prev = new_dp
        
        return dp_prev[n]