class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 2:
            return True
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Initialize DP table
        dp = [[False] * n for _ in range(n)]
        
        # Fill DP for subarrays of length 1 and 2
        for i in range(n):
            dp[i][i] = True
            if i < n - 1:
                dp[i][i + 1] = True
        
        # Check subarrays of length >=3
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    left_sum = prefix[k + 1] - prefix[i]
                    right_sum = prefix[j + 1] - prefix[k + 1]
                    left_valid = (k == i) or (left_sum >= m)
                    right_valid = (j == k + 1) or (right_sum >= m)
                    if left_valid and right_valid and dp[i][k] and dp[k + 1][j]:
                        dp[i][j] = True
                        break  # No need to check further splits
        
        return dp[0][n - 1]