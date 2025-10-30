class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        if a == b:
            x = (1 << n) - 1
            return ((a ^ x) * (b ^ x)) % MOD
        
        if n == 0:
            return (a * b) % MOD
        
        ans = 0
        
        for i in range(1 << n):
            ans = max(ans, (a ^ i) * (b ^ i))
        
        return ans % MOD