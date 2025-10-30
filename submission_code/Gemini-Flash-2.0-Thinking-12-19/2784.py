class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        MOD = 10**9 + 7
        total_power = 0
        p_sum = 0
        for j in range(n):
            current_val = nums[j]
            power_cube = (current_val * current_val) % MOD
            power_cube = (power_cube * current_val) % MOD
            total_power = (total_power + power_cube) % MOD
            if j > 0:
                term = (current_val * current_val) % MOD
                term = (term * p_sum) % MOD
                total_power = (total_power + term) % MOD
            p_sum = (p_sum * 2) % MOD
            p_sum = (p_sum + nums[j]) % MOD
        return total_power