from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        MAX_VAL = 50 # Maximum possible value of nums[i] based on constraints

        # dp[j] represents the number of valid arr1 sequences of length i ending with arr1[i] = j
        # We use two rows to save space: prev_dp and curr_dp
        # The size of the DP array is based on the maximum possible value of arr1[i], which is nums[i] <= MAX_VAL.
        
        # Base case i = 0
        # arr1[0] can be any integer from 0 to nums[0]
        prev_dp = [0] * (MAX_VAL + 1)
        for j in range(nums[0] + 1):
            prev_dp[j] = 1

        # DP transitions for i from 1 to n-1
        for i in range(1, n):
            # Compute prefix sums for prev_dp
            # prev_S[x] = sum(prev_dp[k] for k in range(x+1))
            prev_S = [0] * (MAX_VAL + 1)
            prev_S[0] = prev_dp[0]
            for x in range(1, MAX_VAL + 1):
                prev_S[x] = (prev_S[x - 1] + prev_dp[x]) % MOD

            curr_dp = [0] * (MAX_VAL + 1)
            
            # Calculate the minimum difference requirement derived from monotonicity constraints
            # arr1[i] - arr1[i-1] >= max(0, nums[i] - nums[i-1])
            # Let arr1[i-1] = k, arr1[i] = j
            # j - k >= max(0, nums[i] - nums[i-1])
            # This is equivalent to k <= j - max(0, nums[i] - nums[i-1])
            min_diff = max(0, nums[i] - nums[i-1])

            # For each possible value j for arr1[i]
            # j must be between 0 and nums[i] based on arr1[i] >= 0 and arr2[i] = nums[i] - arr1[i] >= 0
            for j in range(nums[i] + 1):
                # We need to sum the number of valid previous states ending at k (dp[i-1][k])
                # where k satisfies the constraints:
                # 1. 0 <= k <= nums[i-1] (k is a valid value for arr1[i-1])
                # 2. k <= j - min_diff (monotonicity constraint derived above)
                # So, k must be in the range [0, min(nums[i-1], j - min_diff)]
                
                K_upper = min(nums[i-1], j - min_diff)

                # The number of valid previous states ending at k is sum(prev_dp[k] for k in range(0, K_upper + 1))
                # This sum can be efficiently computed using the prefix sum array prev_S
                # sum(prev_dp[k] for k in range(0, K_upper + 1)) = prev_S[K_upper]
                # We need to handle the case where K_upper is negative, meaning the upper bound is less than 0.
                # In that case, the range [0, K_upper + 1] is empty or starts <= 0 and ends < 0, so the sum is 0.
                if K_upper >= 0:
                    # Access prev_S at index K_upper.
                    # K_upper is at most min(nums[i-1], j - min_diff) <= min(50, 50 - 0) = 50.
                    # So index K_upper is within the valid range [0, 50] for prev_S.
                    curr_dp[j] = prev_S[K_upper]
                # If K_upper < 0, the range is empty, the sum is 0. curr_dp[j] is already initialized to 0.

            # Update prev_dp for the next iteration
            prev_dp = curr_dp

        # The total count is the sum of dp[n-1][j] for all valid j
        # Valid j for the last element arr1[n-1] are 0 <= j <= nums[n-1]
        total_count = 0
        for j in range(nums[n-1] + 1):
            total_count = (total_count + prev_dp[j]) % MOD

        return total_count