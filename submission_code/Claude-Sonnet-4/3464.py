class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # dp[i][0] = max cost ending at i where nums[i] has positive sign
        # dp[i][1] = max cost ending at i where nums[i] has negative sign
        
        # Initialize for first element (always positive in its subarray)
        dp_pos = nums[0]  # dp[0][0]
        dp_neg = float('-inf')  # dp[0][1] - impossible
        
        for i in range(1, n):
            new_dp_pos = max(
                nums[i],  # start new subarray at i
                dp_neg + nums[i]  # continue from previous negative
            )
            new_dp_neg = dp_pos - nums[i]  # continue from previous positive
            
            dp_pos = new_dp_pos
            dp_neg = new_dp_neg
        
        return max(dp_pos, dp_neg)