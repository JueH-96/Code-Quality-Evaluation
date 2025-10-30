class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        prefix = [1] * n
        suffix = [1] * n
        
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * 2 % MOD
        
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] * 2 % MOD
        
        total_power = 0
        for i in range(n):
            left_combinations = prefix[i]
            right_combinations = suffix[i]
            total_combinations = left_combinations * right_combinations % MOD
            total_power += nums[i] ** 2 * total_combinations % MOD
        
        return total_power % MOD