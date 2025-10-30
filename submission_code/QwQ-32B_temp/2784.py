class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        prefix_sum_min = [0] * (n + 1)
        for m in range(1, n + 1):
            prefix_sum_min[m] = (2 * prefix_sum_min[m - 1] + nums[m - 1]) % MOD
        
        total = 0
        for i in range(n):
            a = nums[i]
            s_i = (a + prefix_sum_min[i]) % MOD
            contribution = ((a * a) % MOD) * s_i % MOD
            total = (total + contribution) % MOD
        
        return total