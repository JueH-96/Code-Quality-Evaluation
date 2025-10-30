class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        @cache
        def dp(i, k):
            if k == 0:
                return 1
            if i >= len(nums):
                return 0
            return dp(i + 1, k) + dp(i + 1, k - nums[i]) * pow(2, i, 10**9 + 7)
        
        return (dp(0, k) - 1) % (10**9 + 7)