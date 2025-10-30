class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # dp[i][j] stores the XOR score of subarray nums[i..j]
        dp = [[0] * n for _ in range(n)]
        
        # Initialize base cases (single elements)
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill the dp table
        # length represents the length of subarray
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j-1] ^ dp[i+1][j]
        
        # max_dp[i][j] stores the maximum XOR score among all subarrays in nums[i..j]
        max_dp = [[0] * n for _ in range(n)]
        
        # Initialize base cases
        for i in range(n):
            max_dp[i][i] = nums[i]
        
        # Fill the max_dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Maximum is either the current subarray's XOR score,
                # or the maximum from smaller ranges
                max_dp[i][j] = max(dp[i][j], max_dp[i][j-1], max_dp[i+1][j])
        
        # Answer queries
        result = []
        for l, r in queries:
            result.append(max_dp[l][r])
        
        return result