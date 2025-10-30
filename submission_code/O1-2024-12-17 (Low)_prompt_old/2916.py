class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # If the array length is 1, no split is needed; trivially true.
        if n == 1:
            return True

        # Precompute prefix sums for quick range-sum lookups
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Function to get the sum of the subarray nums[l..r]
        def subarray_sum(l, r):
            return prefix[r + 1] - prefix[l]
        
        # dp[i][j] will be True if we can eventually split nums[i..j]
        # into single-element arrays according to the rules.
        dp = [[False] * n for _ in range(n)]
        
        # Base case: subarrays of length 1
        for i in range(n):
            dp[i][i] = True
        
        # Consider subarray lengths from 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Try all possible splits between i..j
                for k in range(i, j):
                    # Check if (nums[i..k]) and (nums[k+1..j]) can be validly produced
                    left_sum = subarray_sum(i, k)
                    right_sum = subarray_sum(k + 1, j)
                    
                    # Left part is valid at the moment of splitting if sum >= m or length=1
                    # Right part is valid in the same way.
                    left_valid_split = (left_sum >= m) or (k - i + 1 == 1)
                    right_valid_split = (right_sum >= m) or (j - (k + 1) + 1 == 1)
                    
                    # If the immediate split is valid, we then check if each part
                    # can eventually be split down to single elements (dp[i][k], dp[k+1][j]).
                    if left_valid_split and right_valid_split and dp[i][k] and dp[k + 1][j]:
                        dp[i][j] = True
                        break
        
        return dp[0][n - 1]