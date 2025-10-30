class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        res = 0
        dp = 0
        for x in nums:
            contribution = (x * x) % MOD
            contribution = contribution * ((dp + x) % MOD) % MOD
            res = (res + contribution) % MOD
            dp = (2 * dp + x) % MOD
        return res % MOD