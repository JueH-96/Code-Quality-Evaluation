class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        inv2 = pow(2, mod - 2, mod)
        n = len(nums)
        f = [0] * (k + 1)
        f[0] = 1
        for x in nums:
            if x <= k:
                for s in range(k, x - 1, -1):
                    f[s] = (f[s] + f[s - x] * inv2) % mod
        total = (f[k] * pow(2, n, mod)) % mod
        return total