from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # Handle the case where the array has only one element
        if n == 1:
            return True
        
        # Compute prefix sums for efficient range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Initialize DP table where dp[i][j] indicates if nums[i..j] can be split validly
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True  # Single element subarrays are valid
        
        # Fill the DP table for subarrays of length >= 2
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Check all possible split points k in [i, j-1]
                for k in range(i, j):
                    # Calculate sums of left and right subarrays
                    sum_left = prefix[k + 1] - prefix[i]
                    sum_right = prefix[j + 1] - prefix[k + 1]
                    
                    # Check if left and right subarrays meet the conditions
                    left_len = k - i + 1
                    right_len = j - (k + 1) + 1
                    left_ok = (left_len == 1) or (sum_left >= m)
                    right_ok = (right_len == 1) or (sum_right >= m)
                    
                    if left_ok and right_ok:
                        # Check if both subarrays can be split validly
                        if dp[i][k] and dp[k + 1][j]:
                            dp[i][j] = True
                            break  # No need to check other splits once a valid one is found
                        
        return dp[0][n - 1]