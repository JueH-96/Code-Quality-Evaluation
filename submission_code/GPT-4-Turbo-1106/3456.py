class Solution:
    def maximumLength(self, nums: list[int], k: int) -> int:
        n = len(nums)
        if k >= n - 1:
            return n
        
        # Initialize the dp array
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Base case: the first element always forms a subsequence of length 1
        for j in range(k + 1):
            dp[0][j] = 1
        
        # Fill the dp array
        for i in range(1, n):
            for j in range(k + 1):
                # Case 1: Include nums[i] and it does not increase the count of changes
                if j > 0 and nums[i] != nums[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = dp[i - 1][j] + 1
                
                # Case 2: Exclude nums[i]
                dp[i][j] = max(dp[i][j], dp[i - 1][j])
        
        return max(dp[-1])

# Example usage:
# sol = Solution()
# print(sol.maximumLength([1,2,1,1,3], 2))  # Output: 4
# print(sol.maximumLength([1,2,3,4,5,1], 0))  # Output: 2