class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        """
        Counts the number of monotonic pairs (arr1, arr2) based on the input array nums.
        A pair is monotonic if arr1 is non-decreasing, arr2 is non-increasing,
        and arr1[i] + arr2[i] == nums[i] for all i.
        """
        MOD = 10**9 + 7
        n = len(nums)

        # max_val is the maximum possible value for any arr1[i], based on constraints.
        # 1 <= nums[i] <= 50
        max_val = 50

        # dp[j] stores the number of valid prefixes arr1[0...i]
        # such that arr1[i] = j. We use a space-optimized DP approach.
        # Let's initialize `dp` for the state at i = 0.
        dp = [0] * (max_val + 1)
        for j in range(nums[0] + 1):
            dp[j] = 1

        # Iterate from the second element (i=1) to the end
        for i in range(1, n):
            # At the beginning of this loop, `dp` holds the counts for index i-1.

            # Calculate prefix sums for dp (state at i-1) to allow for O(1) range sums.
            prefix_sum = [0] * (max_val + 1)
            prefix_sum[0] = dp[0]
            for k in range(1, max_val + 1):
                prefix_sum[k] = (prefix_sum[k - 1] + dp[k]) % MOD

            # Compute the new dp table for the state at index i
            next_dp = [0] * (max_val + 1)
            
            num_prev = nums[i - 1]
            num_curr = nums[i]

            # arr1[i] can range from 0 to nums[i]
            for j in range(num_curr + 1):
                # We need to find the number of valid arr1[i-1] values.
                # Let k = arr1[i-1]. The following conditions must hold for k:
                # 1. 0 <= k <= num_prev (as arr1[i-1] + arr2[i-1] = num_prev)
                # 2. k <= j (as arr1 is non-decreasing)
                # 3. nums[i-1] - k >= nums[i] - j => k <= j + num_prev - num_curr (as arr2 is non-increasing)
                #
                # Combining these, k must be in the range [0, upper_k].
                upper_k = min(num_prev, j, j + num_prev - num_curr)
                
                # The number of valid prefixes ending at arr1[i]=j is the sum of ways
                # to form prefixes ending at any valid k. This sum is exactly the prefix_sum.
                if upper_k >= 0:
                    next_dp[j] = prefix_sum[upper_k]
            
            # Update dp for the next iteration
            dp = next_dp

        # The final result is the sum of all values in the last dp table,
        # which represents the total number of valid full arr1 arrays.
        total_count = sum(dp) % MOD
        return total_count