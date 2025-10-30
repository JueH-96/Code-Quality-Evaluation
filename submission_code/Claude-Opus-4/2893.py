class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # dp[0] = max score ending with even number
        # dp[1] = max score ending with odd number
        dp = [-float('inf'), -float('inf')]
        
        # Initialize based on first element
        if nums[0] % 2 == 0:
            dp[0] = nums[0]
        else:
            dp[1] = nums[0]
        
        # Process each position
        for i in range(1, n):
            curr_parity = nums[i] % 2
            
            # New dp values for current position
            new_dp = [-float('inf'), -float('inf')]
            
            # If current number is even
            if curr_parity == 0:
                # Coming from even number (no penalty)
                new_dp[0] = max(new_dp[0], dp[0] + nums[i])
                # Coming from odd number (penalty x)
                new_dp[0] = max(new_dp[0], dp[1] + nums[i] - x)
            else:
                # If current number is odd
                # Coming from odd number (no penalty)
                new_dp[1] = max(new_dp[1], dp[1] + nums[i])
                # Coming from even number (penalty x)
                new_dp[1] = max(new_dp[1], dp[0] + nums[i] - x)
            
            # Update dp with best scores including current position
            dp[0] = max(dp[0], new_dp[0])
            dp[1] = max(dp[1], new_dp[1])
        
        return max(dp[0], dp[1])