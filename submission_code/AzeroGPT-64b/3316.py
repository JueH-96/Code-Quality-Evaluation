class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        nums.sort()
        mod, n = 10**9 + 7, len(nums)

        def comboFormula(num, k): return math.comb(num, k)

        def powerFormula(num, k): return (num[-k] - num[0]) * comboFormula(n - 1, k - 1)

        def spaceSumFormula(num, k):
            s, s2 = 0, 0
            for j in range(1, len(num)):
                s += num[j] - num[j - 1]
                s = s % mod
                s2 += s * comboFormula(n - 2, k - 1)
                s2 = s2 % mod
            return s2

        ans = 0

        for j in range(k, n):
            val = powerFormula(nums[:j + 1], k)
            val = (val + ((j - k + 1) * spaceSumFormula(nums[:j + 1], k)) % mod) % mod
            ans += val
            ans %= mod

        return ans