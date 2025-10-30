class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        MOD = 10**9 + 7
        total_sum = 0
        n = len(nums)
        for i in range(n):
            total_sum += (nums[i]**2 * nums[i] * (2*i - n + 1)) % MOD
        return total_sum % MOD