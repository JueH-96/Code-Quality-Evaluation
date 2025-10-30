class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # Edge case: if n == 1, we already have 1 subarray
        if n == 1:
            return True
        
        # Precompute prefix sums for O(1) subarray sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # dp[i][j] indicates whether we can split the subarray nums[i..j]
        # into single elements under the given rule
        dp = [[False] * n for _ in range(n)]
        
        # Base case: subarray of length 1 can always be considered 'split'
        for i in range(n):
            dp[i][i] = True
        
        # Helper function to compute sum of subarray nums[i..j]
        def subarray_sum(i, j):
            return prefix[j + 1] - prefix[i]
        
        # Fill dp for subarrays of length from 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Try splitting the subarray [i..j] into [i..k] and [k+1..j]
                for k in range(i, j):
                    left_len = k - i + 1
                    right_len = j - (k + 1) + 1
                    
                    # Check if both halves are valid per step requirement
                    # and if we can recursively split them as well
                    if dp[i][k] and dp[k+1][j]:
                        left_valid = (left_len == 1) or (subarray_sum(i, k) >= m)
                        right_valid = (right_len == 1) or (subarray_sum(k+1, j) >= m)
                        if left_valid and right_valid:
                            dp[i][j] = True
                            break
        
        return dp[0][n-1]