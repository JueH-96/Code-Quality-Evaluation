class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        if n > 60:
            return 0

        ans = 0
        for x in range(2**n):
            prod = (a ^ x) * (b ^ x)
            ans = max(ans, prod)
        return ans % MOD