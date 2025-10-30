from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        # Sort the array. After sorting, any subset will have its minimum as the smallest
        # selected and maximum as the largest selected.
        nums.sort()
        
        # The idea:
        # For a subset with minimum nums[i] and maximum nums[j] (i < j),
        # any subset of indices between i and j can be arbitrarily chosen.
        # So, count = 2^(j - i - 1). Its contribution: nums[j]^2 * nums[i] * 2^(j-i-1)
        # Also, for subsets with only one element (i=j), contribution = nums[i]^3.
        # Hence, the total answer is:
        #    sum_{i=0}^{n-1} nums[i]^3 + sum_{j=1}^{n-1} [nums[j]^2 * (sum_{i=0}^{j-1} nums[i] * 2^(j-i-1))]
        # We precompute a DP array where dp[k] = sum_{i=0}^{k} nums[i] * 2^(k-i)
        # Then for a fixed j, the sum_{i=0}^{j-1} nums[i] * 2^(j-i-1) is exactly dp[j-1].
        
        dp = [0] * n  # dp[k] = sum_{i=0}^{k} nums[i] * 2^(k-i) modulo mod
        dp[0] = nums[0] % mod
        
        total = (nums[0] * nums[0] % mod) * nums[0] % mod  # This is the contribution for the single element subset for index 0
        
        ans = total  # We will add contributions of single element subsets (nums[i]^3)
        for j in range(1, n):
            # Update dp[j] using recurrence:
            # dp[j] = 2 * dp[j-1] + nums[j]
            dp[j] = (2 * dp[j-1] + nums[j]) % mod
            # Add current single element subset's contribution: nums[j]^3
            ans = (ans + (nums[j] * nums[j] % mod) * nums[j]) % mod
            # Now add contribution from those subsets with maximum = nums[j] and minimum from an earlier index.
            # For fixed j, contribution is: nums[j]^2 * dp[j-1]
            ans = (ans + (nums[j] * nums[j] % mod) * dp[j-1]) % mod
        
        return ans % mod