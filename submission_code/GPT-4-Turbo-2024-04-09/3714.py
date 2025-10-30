class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute powers of 2 up to n, modulo MOD
        power_of_two = [1] * (n + 1)
        for i in range(1, n + 1):
            power_of_two[i] = (power_of_two[i - 1] * 2) % MOD
        
        # Calculate the contribution of each element as minimum and maximum
        total_sum = 0
        for i in range(n):
            # When nums[i] is the maximum in a subsequence
            if i + 1 <= k:
                # All subsequences ending at i are valid
                max_contrib = (nums[i] * power_of_two[i]) % MOD
            else:
                # Only subsequences of length up to k are valid
                max_contrib = (nums[i] * (power_of_two[i] - power_of_two[i - k] + MOD) % MOD) % MOD
            
            # When nums[i] is the minimum in a subsequence
            if n - i <= k:
                # All subsequences starting at i are valid
                min_contrib = (nums[i] * power_of_two[n - i - 1]) % MOD
            else:
                # Only subsequences of length up to k are valid
                min_contrib = (nums[i] * (power_of_two[n - i - 1] - power_of_two[n - i - 1 - k] + MOD) % MOD) % MOD
            
            # Add contributions to total sum
            total_sum = (total_sum + max_contrib + min_contrib) % MOD
        
        return total_sum