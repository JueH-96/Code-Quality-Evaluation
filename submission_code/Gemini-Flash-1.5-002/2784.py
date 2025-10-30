class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums.sort()
        n = len(nums)
        ans = 0
        prefix_prod = 1
        for i in range(n):
            ans = (ans + (nums[i] ** 2) * nums[i] * pow(2, i, mod) * prefix_prod) % mod
            prefix_prod = (prefix_prod * nums[i]) % mod

        
        return ans