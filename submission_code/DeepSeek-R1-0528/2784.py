class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums.sort()
        s = 0
        res = 0
        for x in nums:
            term = (x + s) % mod
            res = (res + (x * x) % mod * term) % mod
            s = (s * 2 + x) % mod
        return res