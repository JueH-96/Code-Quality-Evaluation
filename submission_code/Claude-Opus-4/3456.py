class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp[i][j] = maximum length of subsequence ending at index i with exactly j differences
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Base case: each element by itself forms a subsequence of length 1 with 0 differences
        for i in range(n):
            dp[i][0] = 1
        
        # For each position
        for i in range(1, n):
            # For each number of differences allowed
            for j in range(k + 1):
                # Option 1: Start a new subsequence at position i
                dp[i][j] = 1
                
                # Option 2: Extend previous subsequences
                for prev in range(i):
                    if nums[prev] == nums[i]:
                        # Same value, no additional difference needed
                        dp[i][j] = max(dp[i][j], dp[prev][j] + 1)
                    elif j > 0:
                        # Different value, use one difference
                        dp[i][j] = max(dp[i][j], dp[prev][j-1] + 1)
        
        # Find the maximum length across all ending positions and all allowed differences
        result = 0
        for i in range(n):
            for j in range(k + 1):
                result = max(result, dp[i][j])
        
        return result