class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # dp[i] will store the maximum score we can get starting from index i
        dp = [0] * n
        dp[0] = nums[0]  # Starting point score
        
        # We will use two variables to track the best scores for even and odd parities
        max_even = -float('inf')
        max_odd = -float('inf')
        
        # Initialize the first element based on its parity
        if nums[0] % 2 == 0:
            max_even = nums[0]
        else:
            max_odd = nums[0]
        
        for i in range(1, n):
            if nums[i] % 2 == 0:
                # If current number is even
                if max_even != -float('inf'):
                    dp[i] = max(dp[i], max_even + nums[i])
                if max_odd != -float('inf'):
                    dp[i] = max(dp[i], max_odd + nums[i] - x)
                # Update max_even
                max_even = max(max_even, dp[i])
            else:
                # If current number is odd
                if max_odd != -float('inf'):
                    dp[i] = max(dp[i], max_odd + nums[i])
                if max_even != -float('inf'):
                    dp[i] = max(dp[i], max_even + nums[i] - x)
                # Update max_odd
                max_odd = max(max_odd, dp[i])
        
        # The answer is the maximum score we can get at the last position
        return max(dp)