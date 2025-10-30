class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        dp = {}
        
        # Sort to process elements in increasing order
        for num in sorted(nums):
            # Calculate both options before updating
            # Option 1: Use num as is, extending sequence ending at num-1
            option1 = dp.get(num - 1, 0) + 1
            # Option 2: Increment num to num+1, extending sequence ending at num
            option2 = dp.get(num, 0) + 1
            
            # Update dp with the best option for each value
            dp[num] = max(dp.get(num, 0), option1)
            dp[num + 1] = max(dp.get(num + 1, 0), option2)
        
        return max(dp.values())