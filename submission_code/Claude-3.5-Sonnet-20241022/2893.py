class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # dp[i][p] represents max score ending at index i with parity p
        dp = [[-float('inf')] * 2 for _ in range(n)]
        
        # Initialize first position
        dp[0][nums[0] % 2] = nums[0]
        
        # For each position
        for i in range(1, n):
            curr_parity = nums[i] % 2
            
            # For previous position with same parity
            dp[i][curr_parity] = nums[i] + max(
                # From same parity
                dp[i-1][curr_parity],
                # From different parity - penalty x
                dp[i-1][1-curr_parity] - x if dp[i-1][1-curr_parity] != float('-inf') else float('-inf')
            )
            
            # For previous position with different parity
            dp[i][1-curr_parity] = dp[i-1][1-curr_parity]
        
        # Return max of last position with either parity
        return max(dp[n-1][0], dp[n-1][1])