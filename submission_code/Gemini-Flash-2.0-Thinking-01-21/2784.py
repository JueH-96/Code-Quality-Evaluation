class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        total_power_sum = 0
        prefix_sum_p = 0
        mod_val = 10**9 + 7
        for i in range(n):
            term1 = pow(nums[i], 3, mod_val)
            term2 = (pow(nums[i], 2, mod_val) * prefix_sum_p) % mod_val
            current_power = (term1 + term2) % mod_val
            total_power_sum = (total_power_sum + current_power) % mod_val
            prefix_sum_p = (2 * prefix_sum_p + nums[i]) % mod_val
        return total_power_sum