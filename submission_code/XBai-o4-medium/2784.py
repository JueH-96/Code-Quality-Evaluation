class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        dp_sum_min = 0
        total = 0
        for num in nums:
            contrib = (num * num) % MOD
            contrib *= (dp_sum_min + num) % MOD
            contrib %= MOD
            total = (total + contrib) % MOD
            dp_sum_min = (2 * dp_sum_min + num) % MOD
        return total