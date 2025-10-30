class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        dp = {}
        
        for num in nums:
            if num not in dp:
                dp[num] = 1
            if num + 1 not in dp:
                dp[num + 1] = 1
            if num - 1 in dp:
                dp[num] = max(dp[num], dp[num - 1] + 1)
            if num in dp:
                dp[num + 1] = max(dp[num + 1], dp[num] + 1)
        
        max_length = 0
        for key in dp:
            if dp[key] > max_length:
                max_length = dp[key]
        
        return max_length