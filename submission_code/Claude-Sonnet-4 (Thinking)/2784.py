class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Compute suffix_sum[i] = sum over j from i to n-1 of nums[j]^2 * 2^(j-i)
        suffix_sum = [0] * (n + 1)  # suffix_sum[n] = 0
        suffix_sum[n - 1] = pow(nums[n - 1], 2, MOD)
        for i in range(n - 2, -1, -1):
            suffix_sum[i] = (pow(nums[i], 2, MOD) + 2 * suffix_sum[i + 1]) % MOD
        
        total = 0
        for i in range(n):
            contribution = nums[i] * (pow(nums[i], 2, MOD) + suffix_sum[i + 1]) % MOD
            total = (total + contribution) % MOD
        
        return total