from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        total_sum = 0
        prefix_sum = 0
        for num in nums:
            total_sum += (num**2) * (prefix_sum + num) % MOD
            total_sum %= MOD
            prefix_sum = (2 * prefix_sum + num) % MOD
        return total_sum