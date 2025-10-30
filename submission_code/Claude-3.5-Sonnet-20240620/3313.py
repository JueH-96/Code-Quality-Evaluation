class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        
        for i in range(k, 0, -1):
            new_dp = [float('-inf')] * (n + 1)
            multiplier = (-1) ** (i + 1) * (k - i + 1)
            
            curr_sum = 0
            max_sum = float('-inf')
            
            for j in range(n - 1, -1, -1):
                curr_sum += nums[j] * multiplier
                max_sum = max(max_sum, curr_sum)
                
                new_dp[j] = max(new_dp[j + 1], max_sum + dp[j + 1])
                
                curr_sum = max(curr_sum, 0)
            
            dp = new_dp
        
        return dp[0]