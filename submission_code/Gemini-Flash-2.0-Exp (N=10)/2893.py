class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [float('-inf')] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            dp[i] = nums[i]
            for j in range(i):
                cost = 0
                if nums[i] % 2 != nums[j] % 2:
                    cost = x
                dp[i] = max(dp[i], dp[j] + nums[i] - cost)
        
        return max(dp)