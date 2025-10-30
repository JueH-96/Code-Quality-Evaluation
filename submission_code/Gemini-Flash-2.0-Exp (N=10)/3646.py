class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        dp = [0] * (max_val + 1)
        
        for num in nums:
            if num == 0:
                dp[num] = (dp[num] * 2 + num) % MOD
            else:
                prev_sum = dp[num-1] if num > 0 else 0
                next_sum = dp[num+1] if num < max_val else 0
                dp[num] = (dp[num] * 2 + num + prev_sum + next_sum) % MOD
        
        total_sum = 0
        for val in dp:
            total_sum = (total_sum + val) % MOD
        
        return total_sum