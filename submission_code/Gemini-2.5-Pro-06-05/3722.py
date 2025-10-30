from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Constraints: 1 <= k <= floor(nums.length / m), so k * m <= n.
        # This implies that it's always possible to find k non-overlapping
        # subarrays of length at least m.

        # Calculate prefix sums for efficient subarray sum calculation.
        # prefix[i] stores the sum of nums[0...i-1].
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]

        # We use dynamic programming. Let dp[i][j] be the maximum sum of j
        # non-overlapping subarrays of length at least m, considering the
        # prefix nums[0...i-1].
        #
        # To optimize space, we use two arrays, dp_prev for j-1 and dp_curr for j.

        # For j=0 (0 subarrays), the maximum sum is 0.
        dp_prev = [0] * (n + 1)
        
        neg_inf = float('-inf')

        for j in range(1, k + 1):
            dp_curr = [neg_inf] * (n + 1)
            
            # This term is part of the optimization. It tracks the maximum of
            # (dp_prev[p] - prefix[p]) for valid split points p.
            max_prev_term = neg_inf
            
            # To form j subarrays of length at least m, we need at least j*m elements.
            for i in range(j * m, n + 1):
                # Update max_prev_term for the current i.
                # The newest candidate for the split point p is i-m.
                # The first j-1 subarrays must be in nums[0...p-1],
                # and the j-th subarray is nums[p...i-1].
                # dp_prev[p] gives the max sum for j-1 subarrays in nums[0...p-1].
                max_prev_term = max(max_prev_term, dp_prev[i - m] - prefix[i - m])
                
                # Option 1: A subarray ends at index i-1.
                # The sum is prefix[i] + max_prev_term.
                if max_prev_term != neg_inf:
                    option1 = prefix[i] + max_prev_term
                else:
                    option1 = neg_inf

                # Option 2: No subarray ends at index i-1.
                # The j subarrays are all within nums[0...i-2].
                # The max sum is then dp_curr[i-1].
                option2 = dp_curr[i-1]
                
                dp_curr[i] = max(option1, option2)
            
            # The current DP results become the previous ones for the next iteration.
            dp_prev = dp_curr

        return dp_prev[n]