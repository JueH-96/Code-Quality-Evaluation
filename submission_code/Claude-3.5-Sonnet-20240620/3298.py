class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        max_consecutive = 0
        
        for num in nums:
            dp[num + 1] = dp.get(num, 0) + 1
            dp[num] = dp.get(num - 1, 0) + 1
            max_consecutive = max(max_consecutive, dp[num], dp[num + 1])
        
        return max_consecutive