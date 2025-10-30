class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        
        for x in nums:
            # Option 1: keep this element as x (extends sequence ending at x-1)
            new_dp_x = dp.get(x - 1, 0) + 1
            
            # Option 2: increase this element to x+1 (extends sequence ending at x)
            new_dp_x_plus_1 = dp.get(x, 0) + 1
            
            # Update dp with the better option for each value
            dp[x] = max(dp.get(x, 0), new_dp_x)
            dp[x + 1] = max(dp.get(x + 1, 0), new_dp_x_plus_1)
        
        return max(dp.values()) if dp else 0