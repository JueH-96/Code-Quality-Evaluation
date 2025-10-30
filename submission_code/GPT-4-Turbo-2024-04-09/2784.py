class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute powers of 2 up to n (since there are at most n elements)
        power_of_two = [1] * (n + 1)
        for i in range(1, n + 1):
            power_of_two[i] = (power_of_two[i - 1] * 2) % MOD
        
        result = 0
        # For each element in nums, calculate its contribution as min and max
        for i in range(n):
            # When nums[i] is the maximum
            # All subsets from start to i where nums[i] is the maximum
            # There are 2^i subsets including empty set, so 2^i - 1 non-empty subsets
            max_contrib = nums[i] * nums[i] * nums[i] % MOD
            max_contrib = max_contrib * (power_of_two[i] - 1) % MOD
            
            # When nums[i] is the minimum
            # All subsets from i to end where nums[i] is the minimum
            # There are 2^(n-i-1) subsets including empty set, so 2^(n-i-1) - 1 non-empty subsets
            min_contrib = nums[i] * nums[n-1] * nums[n-1] % MOD
            min_contrib = min_contrib * (power_of_two[n-i-1] - 1) % MOD
            
            # Add contributions to result
            result = (result + max_contrib + min_contrib) % MOD
        
        return result