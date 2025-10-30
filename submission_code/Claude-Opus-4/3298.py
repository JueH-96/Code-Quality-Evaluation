class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Sort the array to process elements in order
        nums.sort()
        
        # dp[x] = length of longest consecutive sequence ending at x
        dp = {}
        
        for num in nums:
            # Option 1: Use num as is
            # The sequence ending at num extends the sequence ending at num-1
            dp[num] = dp.get(num - 1, 0) + 1
            
            # Option 2: Use num + 1
            # The sequence ending at num+1 extends the sequence ending at num
            dp[num + 1] = dp.get(num, 0) + 1
        
        # Return the maximum length found
        return max(dp.values())