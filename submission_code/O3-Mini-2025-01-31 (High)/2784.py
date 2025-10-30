class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Sort the heroes by strength.
        nums.sort()
        
        ans = 0
        # First, add the contribution of single-element groups.
        # For a single hero x, the power is x^2 * x = x^3.
        for x in nums:
            ans = (ans + (x % MOD) * (x % MOD) % MOD * (x % MOD)) % MOD
        
        # Now, consider groups with at least 2 elements.
        # For any nonempty subset, if we fix the minimum as nums[i] and maximum as nums[j] (with i < j),
        # then all heroes chosen from indices (i+1, ..., j-1) can be either chosen or not.
        # So, for the pair (i, j), there are 2^(j-i-1) possibilities.
        # The power contribution of each such group is nums[j]^2 * nums[i].
        # Hence, for a fixed j (the maximum) the sum of contributions is:
        #   nums[j]^2 * ( sum_{i=0}^{j-1} nums[i]*(2^(j-i-1)) ).
        #
        # We compute f(j) = sum_{i=0}^{j-1} nums[i] * 2^(j-i-1)
        # Notice the recurrence:
        #   f(1) = nums[0]
        #   f(j+1) = 2 * f(j) + nums[j]
        
        f = nums[0] % MOD  # This is f(1)
        for j in range(1, n):
            # contribution for groups that have nums[j] as the maximum
            ans = (ans + (nums[j] % MOD) * (nums[j] % MOD) % MOD * f) % MOD
            # Update f to get f(j+1)
            f = (2 * f + nums[j]) % MOD
        
        return ans % MOD