class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        
        for num in nums:
            # For each number, we have two choices:
            # 1. Keep it as is
            # 2. Increment it by 1
            
            # For original number
            dp[num] = dp.get(num - 1, 0) + 1
            
            # For incremented number
            dp[num + 1] = dp.get(num, 0) + 1
            
        return max(dp.values())