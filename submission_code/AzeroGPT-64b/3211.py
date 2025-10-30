class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        left = 0
        
        for right in range(n):
            while left < right and nums[right] < nums[right - 1]:
                left += 1
            if left < right:
                sum_ = sum(nums[left:right + 1])
                i, j = max(0, left - 1), min(n - 1, right + 1)
                while i >= 0 and nums[i] <= sum_:
                    i -= 1
                while j < n and nums[j] >= sum_:
                    j += 1
                left, right = max(0, i + 1), min(n - 1, j - 1)
            dp[right] = right - max(0, left - 1)
            left = right
        
        return dp[n - 1]