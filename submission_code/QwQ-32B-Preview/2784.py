class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        # Initialize suffix_sum array
        suffix_sum = [0] * (n + 1)
        # Compute suffix_sum from the end
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = (pow(nums[i], 2, MOD) + 2 * suffix_sum[i + 1] % MOD) % MOD
        # Compute total sum of powers
        total_sum = 0
        for i in range(n):
            contribution = (pow(nums[i], 3, MOD) + nums[i] * suffix_sum[i + 1] % MOD) % MOD
            total_sum = (total_sum + contribution) % MOD
        return total_sum