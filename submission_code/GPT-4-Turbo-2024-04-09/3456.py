class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k == 0:
            # When k is 0, we can only have sequences of identical elements
            max_len = 1
            current_len = 1
            for i in range(1, n):
                if nums[i] == nums[i-1]:
                    current_len += 1
                else:
                    max_len = max(max_len, current_len)
                    current_len = 1
            max_len = max(max_len, current_len)
            return max_len
        
        # dp[i][j] will be the maximum length of a good subsequence ending at index i with j changes
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Initialize for the first element
        for j in range(k + 1):
            dp[0][j] = 1
        
        # Fill the dp table
        for i in range(1, n):
            for j in range(k + 1):
                # Continue the sequence without a change
                if nums[i] == nums[i-1]:
                    dp[i][j] = dp[i-1][j] + 1
                else:
                    # We need to make a change
                    if j > 0:
                        dp[i][j] = dp[i-1][j-1] + 1
                
                # Check if we can start a new sequence here
                dp[i][j] = max(dp[i][j], 1)
                
                # Also consider extending the sequence from any previous point with the same value
                for m in range(i):
                    if nums[m] == nums[i] and dp[m][j] > 0:
                        dp[i][j] = max(dp[i][j], dp[m][j] + 1)
        
        # Find the maximum value in dp table
        max_length = 0
        for i in range(n):
            for j in range(k + 1):
                max_length = max(max_length, dp[i][j])
        
        return max_length