class Solution:
    MOD = 10**9 + 7

    def minMaxSums(self, nums, k: int) -> int:
        nums.sort()
        n = len(nums)
        total_min = 0
        total_max = 0

        for i in range(n):
            # Calculate contribution to minimum sum
            a_min = n - i - 1
            t_max_min = min(k - 1, a_min)
            sum_min = self.compute_combination_sum(a_min, t_max_min)
            total_min = (total_min + nums[i] * sum_min) % self.MOD

            # Calculate contribution to maximum sum
            a_max = i
            t_max_max = min(k - 1, a_max)
            sum_max = self.compute_combination_sum(a_max, t_max_max)
            total_max = (total_max + nums[i] * sum_max) % self.MOD

        return (total_min + total_max) % self.MOD

    def compute_combination_sum(self, a, t_max):
        if t_max < 0:
            return 0
        total = 1
        current = 1
        for t in range(1, t_max + 1):
            numerator = current * (a - t + 1)
            inv_t = pow(t, self.MOD - 2, self.MOD)
            current = (numerator * inv_t) % self.MOD
            total = (total + current) % self.MOD
        return total