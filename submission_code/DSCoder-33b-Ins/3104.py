from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        fact = [1] * (n + 1)
        invfact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % mod
            invfact[i] = pow(fact[i], mod - 2, mod)
        def nCr(n, r):
            if r > n:
                return 0
            return fact[n] * invfact[r] * invfact[n - r] % mod
        nums.sort()
        ans = 1
        for i in range(n):
            ans = ans * nCr(nums[i] - (0 if i == 0 else nums[i - 1]), nums[i] - i) % mod
        return ans