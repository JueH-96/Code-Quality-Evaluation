class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        res = 0
        s = 0
        mod = 10**9 + 7
        for j in range(len(nums)):
            res = (res + nums[j] * nums[j] * (s + nums[j])) % mod
            s = (s * 2 + nums[j]) % mod
        return res