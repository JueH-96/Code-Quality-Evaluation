class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}  # dp[x] = length of consecutive sequence ending at x
        
        for num in nums:
            # Option 1: keep num as-is
            dp[num] = dp.get(num - 1, 0) + 1
            
            # Option 2: increase num by 1
            dp[num + 1] = dp.get(num, 0) + 1
        
        return max(dp.values())