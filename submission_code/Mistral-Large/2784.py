from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        prefix_square_sum = [0] * (n + 1)

        for i in range(1, n + 1):
            prefix_sum[i] = (prefix_sum[i - 1] + nums[i - 1]) % MOD
            prefix_square_sum[i] = (prefix_square_sum[i - 1] + nums[i - 1] * nums[i - 1]) % MOD

        total_power = 0
        for i in range(n):
            max_val = nums[i]
            min_val_sum = (max_val * (i + 1) - prefix_sum[i] + MOD) % MOD
            max_square_sum = (max_val * max_val * (i + 1) - prefix_square_sum[i] + MOD) % MOD
            total_power = (total_power + max_val * max_square_sum % MOD * min_val_sum % MOD) % MOD

        return total_power