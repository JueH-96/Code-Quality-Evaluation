class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        nums.sort()
        ans = 0
        prefix_sum = 0
        for i in range(n):
            ans = (ans + (nums[i] * nums[i] % mod) * (nums[i] + prefix_sum) % mod) % mod
            prefix_sum = (prefix_sum * 2 + nums[i]) % mod
        return ans