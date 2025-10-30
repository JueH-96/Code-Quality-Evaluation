class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # dp[0] = max score ending at a position with even number
        # dp[1] = max score ending at a position with odd number
        dp = [-float('inf')] * 2
        
        # Initialize with the first element
        dp[nums[0] % 2] = nums[0]
        
        # Process each position starting from index 1
        for i in range(1, n):
            parity = nums[i] % 2
            
            # Option 1: Come from same parity (no penalty)
            same_parity_score = dp[parity] + nums[i]
            
            # Option 2: Come from different parity (penalty of x)
            diff_parity_score = dp[1 - parity] + nums[i] - x
            
            # Take the maximum of both options
            dp[parity] = max(same_parity_score, diff_parity_score)
        
        # Return the maximum score we can achieve
        return max(dp[0], dp[1])