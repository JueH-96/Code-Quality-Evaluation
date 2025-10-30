class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        mod = 10**9 + 7
        total_power = 0

        for i in range(n):
            power_single = pow(nums[i], 3, mod)
            total_power = (total_power + power_single) % mod

            for j in range(i + 1, n):
                num_groups = pow(2, j - i - 1, mod)
                power = (pow(nums[j], 2, mod) * nums[i]) % mod
                power = (power * num_groups) % mod
                total_power = (total_power + power) % mod

        return total_power