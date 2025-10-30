class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        sum_cubes = 0
        for x in sorted_nums:
            sum_cubes = (sum_cubes + pow(x, 3, MOD)) % MOD
        pre_sum = [0] * n
        for j in range(1, n):
            pre_sum[j] = (2 * pre_sum[j-1] + sorted_nums[j-1]) % MOD
        sum_contrib = 0
        for j in range(n):
            x = sorted_nums[j]
            contrib = (pow(x, 2, MOD) * pre_sum[j]) % MOD
            sum_contrib = (sum_contrib + contrib) % MOD
        total = (sum_cubes + sum_contrib) % MOD
        return total