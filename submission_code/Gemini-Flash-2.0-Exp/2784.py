class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        mod = 10**9 + 7
        prefix_sum = 0
        for i in range(n):
            ans = (ans + (nums[i]**2) * (nums[i] + prefix_sum)) % mod
            prefix_sum = (prefix_sum * 2 + nums[i]) % mod
        return ans