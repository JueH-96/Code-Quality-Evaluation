class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # dp[i][0] = max cost up to index i where nums[i] has positive sign
        # dp[i][1] = max cost up to index i where nums[i] has negative sign
        
        # We can optimize space by only keeping track of previous state
        # prev_pos = max cost where previous element had positive sign
        # prev_neg = max cost where previous element had negative sign
        
        prev_pos = nums[0]
        prev_neg = float('-inf')  # Can't start with negative sign
        
        for i in range(1, n):
            curr_pos = max(prev_pos, prev_neg) + nums[i]  # Start new subarray
            curr_neg = prev_pos - nums[i]  # Continue subarray with negative sign
            
            prev_pos = curr_pos
            prev_neg = curr_neg
        
        return max(prev_pos, prev_neg)