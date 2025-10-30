class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = nums[0]
        even = nums[0] if nums[0] % 2 == 0 else 0
        odd = nums[0] if nums[0] % 2 != 0 else 0
        
        for i in range(1, n):
            if nums[i] % 2 == 0:
                even = max(even + nums[i], odd + nums[i] - x)
            else:
                odd = max(odd + nums[i], even + nums[i] - x)
            dp[i] = max(even, odd)
            
        return dp[n - 1]