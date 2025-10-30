class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7

        if n <= 60:
            ans = 0
            for x in range(2**n):
                ans = max(ans, (a ^ x) * (b ^ x))
            return ans % MOD

        ans = 0
        for x in range(1 << n):
            ans = max(ans, (a ^ x) * (b ^ x))
        return ans % (10**9 + 7)