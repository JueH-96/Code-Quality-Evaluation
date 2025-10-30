class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        total_sum = 0
        prefix_sum = 0

        for i in range(len(nums)):
            total_sum = (total_sum + nums[i] * nums[i] * (prefix_sum + nums[i])) % MOD
            prefix_sum = (2 * prefix_sum + nums[i]) % MOD

        return total_sum