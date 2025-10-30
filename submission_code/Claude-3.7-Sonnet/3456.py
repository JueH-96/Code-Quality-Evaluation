class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][j] = max length of subsequence ending at index i with exactly j differences
        dp = [[-1 for _ in range(k + 1)] for _ in range(n)]
        
        # Base case: a subsequence of length 1 has 0 differences
        for i in range(n):
            dp[i][0] = 1
        
        # Fill the dp table
        for i in range(n):
            for j in range(k + 1):
                # Consider all possible previous positions
                for p in range(i):
                    # If we add nums[i] after nums[p], check if it creates a difference
                    diff = 1 if nums[i] != nums[p] else 0
                    
                    # Only update if we have a valid previous state and don't exceed allowed differences
                    if j >= diff and dp[p][j - diff] != -1:
                        dp[i][j] = max(dp[i][j], dp[p][j - diff] + 1)
        
        # Find the maximum length across all ending positions and valid difference counts
        max_length = 0
        for i in range(n):
            for j in range(k + 1):
                if dp[i][j] != -1:
                    max_length = max(max_length, dp[i][j])
        
        return max_length